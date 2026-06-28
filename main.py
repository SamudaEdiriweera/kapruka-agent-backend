"""
What this file does

This is the entry point of the entire backend. 
It creates the FastAPI app, registers middleware, 
and mounts the router. This is what uvicorn runs.

Why we need it

FastAPI needs a single app instance to attach everything to — CORS, routes, startup config all live here.

"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from dotenv import load_dotenv

load_dotenv()

app =FastAPI(
    title="kapruka-agent-backend",
    description="Kapru - Sri Lankan AI shopping agent powered by Kapruka MCP",
    version="1.0.0"
)

# ---------- CORS ------------------
# Allow frontend dev server + production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://kapruka-agent-eosin.vercel.app",   # ← your Vercel frontend
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",   # ← Vercel preview deploys
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------- ROUTES --------------------
app.include_router(router, prefix="/api")

# ------- ROOT -----------------------
@app.get("/")
def root():
    return {
        "agent": "kapru",
        "status": "running",
        "version": "1.0.0"
    }
