# Deployment — Kapri Backend

This document describes how the Kapru backend is built, containerized, and deployed to Google Cloud Run through an automated CI/CD pipeline.

## Architecture Overview

```
Local code  →  GitHub (main)  →  GitHub Actions  →  Artifact Registry  →  Cloud Run
                                       │
                                       ├─ build Docker image
                                       ├─ push to Artifact Registry
                                       └─ deploy to Cloud Run (with secrets)

Frontend (Vercel)  ──HTTPS──>  Cloud Run backend URL
```

The frontend runs on Vercel and calls the backend's public Cloud Run URL. Secrets are stored in Google Secret Manager, never in code or logs.

## Stack

| Layer | Technology |
|---|---|
| Runtime | FastAPI (Python 3.13), uvicorn |
| Container | Docker (multi-stage), `uv` package manager |
| Image registry | Google Artifact Registry |
| Compute | Google Cloud Run (serverless, scale-to-zero) |
| Secrets | Google Secret Manager |
| CI/CD | GitHub Actions |
| Frontend host | Vercel |

## Current Deployment Method (what we used)

### 1. Containerization

A multi-stage Dockerfile builds dependencies with `uv` in a builder stage, then copies the virtual environment into a slim runtime image. The container listens on port 8080 (Cloud Run's default).

Key points:
- `ENV PATH="/app/.venv/bin:$PATH"` so uvicorn is found at runtime
- `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]`
- `.dockerignore` excludes `.env`, `.venv`, `.git`, caches

### 2. One-Time GCP Setup

Enabled APIs:
```
run.googleapis.com
artifactregistry.googleapis.com
cloudbuild.googleapis.com
secretmanager.googleapis.com
iam.googleapis.com
```

Created resources:
- Artifact Registry Docker repository (`kapru`, region `us-central1`)
- Secrets in Secret Manager: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- A deployer service account (`github-deployer`) for GitHub Actions

### 3. Service Accounts and Permissions

Two distinct identities are involved (this is a common gotcha):

| Service account | Role | Purpose |
|---|---|---|
| `github-deployer` | `artifactregistry.writer`, `run.admin`, `secretmanager.secretAccessor`, `iam.serviceAccountUser` | Used by GitHub Actions to build, push, and deploy |
| `<project-number>-compute@developer` | `secretmanager.secretAccessor` | The Cloud Run **runtime** identity that reads secrets while the container runs |

Both need `secretmanager.secretAccessor` — the deployer to configure the secrets, the runtime to read them at request time. Missing the runtime grant causes a "Permission denied on secret" deploy failure.

### 4. GitHub Secrets

Only the genuinely sensitive value lives in GitHub Secrets:
- `GCP_SA_KEY` — the deployer service account JSON key

Non-sensitive config (project ID, region, repo name) is hardcoded in the workflow file, which is clearer and avoids empty-variable bugs.

### 5. The CI/CD Pipeline

The workflow (`.github/workflows/deploy.yml`) triggers on push to `main` and runs two jobs:

1. **build-and-push** — authenticates to GCP, builds the Docker image tagged with the commit SHA, and pushes it to Artifact Registry.
2. **deploy** — deploys the image to Cloud Run, injecting secrets from Secret Manager via `--set-secrets` and the public MCP URL via `--set-env-vars`.

Each image is tagged with the git commit SHA, so every deploy is traceable and any previous version can be redeployed for rollback.

### 6. Frontend Wiring

The frontend reads the backend URL from an environment variable and prefixes all API calls with it:

```
NEXT_PUBLIC_API_BASE_URL = https://<cloud-run-url>
```

The backend's CORS config allows the Vercel frontend origin. The env var name in Vercel must match the name the frontend code reads.

### 7. Keeping It Warm

Cloud Run scales to zero when idle, causing a cold start (~30–60s) on the next request. A free uptime pinger (e.g. UptimeRobot) hitting `/api/health` every 5 minutes keeps the service warm for demos.

## Day-to-Day Workflow

Once the pipeline exists, deploying a change is simply:

```
git checkout dev
# make changes
git add .
git commit -m "your change"
git push origin dev
# open a PR dev -> main, merge it
# merge to main triggers the pipeline -> auto-deploys
```

No manual Docker builds, no gcloud commands, no SSH. The Cloud Run URL stays the same across deploys, so the frontend keeps working.

## Why No nginx?

Cloud Run is fully managed and handles HTTPS termination, routing, and load balancing automatically. nginx is only needed on a raw VM (Compute Engine), where none of that is provided. On Cloud Run it would be redundant.

## Cost

For competition-level traffic, the deployment runs effectively free:
- Cloud Run free tier: 2M requests, 180k vCPU-seconds, 360k GiB-seconds per month
- $300 new-customer credit covers any overflow for 90 days
- The real ongoing cost is the LLM API calls (Claude + Gemini), billed separately

## Rollback

Because every image is tagged with its commit SHA, rolling back is redeploying a previous tag:

```
gcloud run deploy kapru-backend \
  --image=us-central1-docker.pkg.dev/<project>/kapru/kapri-backend:<old-sha> \
  --region=us-central1
```

---

## Next / Future Improvements

The current setup is production-grade for this scale. If the project grows, these are the natural next steps, roughly in order of value.

### 1. Workload Identity Federation (remove the long-lived key)

Currently GitHub Actions authenticates with a service account JSON key stored in GitHub Secrets. This is the standard, accepted approach, but the key is a long-lived credential. Workload Identity Federation lets GitHub Actions authenticate to GCP with short-lived tokens and **no stored key at all** — the more secure, modern pattern. This is the single biggest hardening upgrade.

### 2. Staging environment + promotion

Add a separate Cloud Run service for staging. Push to `main` auto-deploys to staging; promoting to production requires a manual approval gate (GitHub Environments with required reviewers, available on public repos or paid plans). This separates "deployed" from "released" — the standard production pattern.

### 3. Automated tests in the pipeline

Add a test job before build-and-push that runs `pytest`. The pipeline then only deploys if tests pass, catching regressions before they reach production.

### 4. Health-check-gated rollout

Cloud Run supports startup and liveness probes. Wiring `/api/health` as a startup probe ensures a bad revision never receives traffic, and a failed rollout auto-rolls-back.

### 5. Observability

Cloud Run integrates with Cloud Logging and Cloud Monitoring out of the box. Adding structured logging and a dashboard (request latency, error rate, cold-start frequency) gives real production visibility. Pairs well with the LangSmith/MLflow tracing already in the app.

### 6. Custom domain

Map a real domain (e.g. `api.kapri.app`) to the Cloud Run service instead of the `.run.app` URL, with a managed SSL certificate. Cleaner for production and branding.

### 7. Min-instances for zero cold start

For a funded production service, setting `--min-instances=1` eliminates cold starts entirely (at a small always-on cost). For now the free UptimeRobot ping achieves the same effect at no cost.

## Summary

| Aspect | This deployment | Next level |
|---|---|---|
| Auth to GCP | SA JSON key in GitHub Secrets | Workload Identity Federation |
| Environments | Single (production) | Staging + production with approval |
| Tests | Manual | Automated in pipeline |
| Cold start | UptimeRobot ping | Min-instances or probes |
| Domain | `.run.app` URL | Custom domain + managed SSL |
| Observability | Default Cloud Logging | Dashboards + structured logs |