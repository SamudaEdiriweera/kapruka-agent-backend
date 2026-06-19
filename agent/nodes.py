"""
What this file does?
This is the brain of the entire agent.
Five nodes, each with a specific job:

    DETECTOR  -> reads user message, detects language + intent
    PLANNER   -> builds a step-by-step shopping plan
    REASONER  -> decides what to do right now
    TOOL CALLER -> executes the right Kapruka MCP tool
    REFLECTOR -> checks if tool result was good, decides next move
    RESPONDER -> builds structured JSON for the frontend
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ShoppingState
from agent import tools
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


load_dotenv()


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0.7,
# )
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
)


KAPRU_SYSTEM_PROMPT = """You are Kapru, a warm and clever Sri Lankan shopping assistant for Kapruka.

CRITICAL LANGUAGE RULE: Always reply in the EXACT same language and style 
the user wrote in:
- English -> reply in English
- Sinhala script -> reply in Sinhala script
- Sinhala in English letters (Singlish) -> reply the same way
- Tamil script -> reply in Tamil script
- Tamil in English letters (Tanglish) -> reply the same way

Be warm, helpful and local - like a smart friend, not a corporate bot.
Use natural Sri Lankan expressions when appropriate (Aney, Aiyo, Machan etc).
ALWAYS respond with valid JSON only. No markdown. No extra text. No code fences. Just JSON."""


# ── HELPERS ──────────────────────────────────────────────────────────────────

def parse_llm_json(content) -> dict:
    """Safely parse LLM JSON response. Handles both string and list content."""
    if isinstance(content, list):
        content = " ".join([
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        ])

    clean = (
        str(content).strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    return json.loads(clean)


def get_last_user_message(state: ShoppingState) -> str:
    """Get the most recent user message from conversation history."""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_text(content) -> str:
    """Extract plain text from LLM response regardless of type."""
    if isinstance(content, list):
        return " ".join([
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        ]).strip()
    return str(content).strip()


# ── DETECTOR NODE ────────────────────────────────────────────────────────────

async def detector_node(state: ShoppingState) -> ShoppingState:
    """
    Reads the latest user message.
    Detects: language, intent, subject, missing info.
    If we are mid-flow (already collecting price/brand), preserve that missing_info
    instead of re-detecting from scratch.
    """
    last_msg = get_last_user_message(state)
    print(f"\n🔍 [DETECTOR] msg: '{last_msg[:50]}'")


    # If frontend already told us what we're waiting for, keep it.
    incoming_missing = state.get("missing_info", [])
    if incoming_missing:
        # We're mid-conversation collecting info — don't reset.
        return {
            **state,
            "language": state.get("language", "en"),
            "intent": state.get("intent", "search_product"),
            "missing_info": incoming_missing,
        }

    prompt = f"""Analyze this message from a Kapruka shopping user: "{last_msg}"

Return JSON:
{{
    "language": "detect and label as one of: en (English), si (Sinhala script),
                 si-tl (Sinhala in English letters), ta (Tamil script),
                 ta-tl (Tamil in English letters)",
    "intent": "search_product | get_product | list_categories | check_delivery | create_order | track_order | add_to_cart | view_cart | general",
    "subject": "extracted product name or empty string",
    "missing_info": []
}}

Language detection examples:
- "I want a phone" -> en
- "මට phone එකක් ඕනේ" -> si
- "Mata phone ekak oney" -> si-tl
- "எனக்கு போன் வேண்டும்" -> ta
- "Enakku oru phone venum" -> ta-tl

Rules for missing_info:
- intent is search_product AND no budget mentioned -> add "price_range"
- intent is search_product AND no brand mentioned -> add "brand"
- intent is check_delivery or create_order AND no city -> add "city"
- intent is check_delivery or create_order AND no date -> add "delivery_date"
- intent is create_order AND no recipient name -> add "recipient_name"
- intent is create_order AND no recipient phone -> add "recipient_phone"
- intent is create_order AND no sender name -> add "sender_name"
- intent is create_order AND no sender phone -> add "sender_phone"
- Only include genuinely missing fields
- Always put "price_range" before "brand" so price is asked first"""

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    parsed = parse_llm_json(res.content)

    print(f"🔍 [DETECTOR] lang={parsed.get('language')} intent={parsed.get('intent')} missing={parsed.get('missing_info')}")
    return {
        **state,
        "language": parsed.get("language", state.get("language", "en")),
        "intent": parsed.get("intent", "general"),
        "missing_info": parsed.get("missing_info", []),
    }


# ── PLANNER NODE ─────────────────────────────────────────────────────────────

async def planner_node(state: ShoppingState) -> ShoppingState:
    """
    Builds a full step-by-step plan based on intent.
    Only replans if plan is empty or reflection triggered a replan.
    """
    print(f"\n📋 [PLANNER] intent={state['intent']}")
    if state.get("plan") and state.get("reflection") != "replan":
        return state

    prompt = f"""User intent: "{state['intent']}"
Language: "{state['language']}"
Cart: {state['cart']}
Missing info: {state['missing_info']}
Conversation: {state['messages'][-4:]}

Build a minimal shopping plan as a JSON array of action strings.
Available actions: search_product, get_product, list_categories, check_delivery,
create_order, track_order, ask_missing_info, show_cart, respond

Example for "send birthday cake":
["search_product", "show_cards", "ask_missing_info", "check_delivery", "ask_gift_message", "create_order", "show_confirmation"]

Return only the JSON array."""

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    plan = parse_llm_json(res.content)
    print(f"📋 [PLANNER] plan={plan}")

    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "reflection": "",
    }


# ── REASONER NODE ────────────────────────────────────────────────────────────

async def reasoner_node(state: ShoppingState) -> ShoppingState:
    """
    Looks at current plan step + state.
    Decides: call a tool, ask user for missing info, or respond.
    Dynamically generates price brackets from live Kapruka data.
    Caches products to avoid double MCP calls.
    """

    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)

    if plan and 0 <= step_idx < len(plan):
        current_plan_step = plan[step_idx]
    else:
        current_plan_step = "respond"


    last_msg = get_last_user_message(state)
    missing = list(state.get("missing_info", []))
    price_quick_replies = []
    brand_quick_replies = []
    cached_products = state.get("cached_products", [])
    price_range = state.get("price_range", {})
    selected_brand = state.get("selected_brand", "")
    available_brands = state.get("available_brands", [])

    print(f"\n🧠 [REASONER] step='{current_plan_step}' missing={missing}")


    prev_messages = state["messages"]
    asked_budget = any(
        "budget" in m.get("content", "").lower()
        or "price" in m.get("content", "").lower()
        or "lkr" in m.get("content", "").lower()
        for m in prev_messages
        if m.get("role") == "assistant"
    )

    # ── Step 1: Parse price range from the user's reply ────────────
    if not price_range and asked_budget and "price_range" in missing:
        try:
            parse_res = llm.invoke([
                SystemMessage(content=KAPRU_SYSTEM_PROMPT),
                HumanMessage(content=f"""User replied: "{last_msg}"
They were answering a budget/price range question.
Extract min and max price in LKR. Return ONLY a JSON object:
{{"min_price": 0, "max_price": 6000}}
Rules:
- "under/below X" -> min_price 0, max_price X
- "above/over X" -> min_price X, max_price null
- any range "X-Y" or "X to Y" (any dash) -> min_price X, max_price Y
- "k" means thousands: "50k" = 50000
- ignore commas, LKR text, and currency words
- if it is not a price at all -> {{"min_price": null, "max_price": null}}""")
            ])
            parsed_range = parse_llm_json(parse_res.content)
            if parsed_range.get("min_price") is not None:
                price_range = parsed_range
                missing = [m for m in missing if m != "price_range"]
                # Filter cached products by the selected range
                if cached_products:
                    min_p = price_range.get("min_price", 0) or 0
                    max_p = price_range.get("max_price")
                    cached_products = [
                        p for p in cached_products
                        if p.get("price", 0) >= min_p
                        and (max_p is None or p.get("price", 0) <= max_p)
                    ]
        except Exception as e:
            print(f"[PRICE PARSE ERROR] {e}")

    # ── Step 2: Generate price brackets if still needed ────────────
    if "price_range" in missing:
        subject_res = llm.invoke([
            SystemMessage(content=KAPRU_SYSTEM_PROMPT),
            HumanMessage(content=f"""Extract only the product name from: "{last_msg}"
Return plain text only. No explanation."""),
        ])
        subject = extract_text(subject_res.content)

        hints = await tools.get_price_range_hints(subject)
        price_quick_replies = hints.get("brackets", [])

        if hints.get("products"):
            cached_products = hints["products"]

        if not price_quick_replies:
            try:
                fallback_res = llm.invoke([
                    SystemMessage(content=KAPRU_SYSTEM_PROMPT),
                    HumanMessage(content=f"""Generate 4 realistic LKR price brackets
for "{subject}" sold on Kapruka.lk Sri Lanka.
Return a JSON array of 4 strings only.
Example: ["Under LKR 500", "LKR 500-2,000", "LKR 2,000-5,000", "Above LKR 5,000"]"""),
                ])
                price_quick_replies = parse_llm_json(fallback_res.content)
            except Exception:
                price_quick_replies = [
                    "Under LKR 1,000",
                    "LKR 1,000-5,000",
                    "LKR 5,000-15,000",
                    "Above LKR 15,000",
                ]

    # ── Step 3: Brand handling (only after price collected) ────────
    price_collected = bool(price_range.get("min_price") is not None)

    if "brand" in missing and price_collected:
        asked_brand = any(
            "brand" in m.get("content", "").lower()
            or "bakery" in m.get("content", "").lower()
            or "prefer" in m.get("content", "").lower()
            for m in prev_messages
            if m.get("role") == "assistant"
        )

        if asked_brand:
            if last_msg.lower() in [
                "any brand", "search all", "any bakery",
                "best for my budget", "no preference",
            ]:
                selected_brand = ""
            else:
                selected_brand = last_msg.strip()
            missing = [m for m in missing if m != "brand"]

            if cached_products and selected_brand:
                cached_products = [
                    p for p in cached_products
                    if selected_brand.lower()
                    in (p.get("brand", "") + p.get("name", "")).lower()
                ]
        else:
            # Need to ask brand — derive real brands from cached products
            if cached_products:
                available_brands = tools.extract_brands(cached_products)
                brand_quick_replies = available_brands
            else:
                subject_res = llm.invoke([
                    SystemMessage(content=KAPRU_SYSTEM_PROMPT),
                    HumanMessage(content=f"""Extract product name from: "{last_msg}"
Return plain text only."""),
                ])
                subject = extract_text(subject_res.content)
                hints = await tools.get_price_range_hints(subject)
                if hints.get("products"):
                    cached_products = hints["products"]
                    available_brands = tools.extract_brands(cached_products)
                    brand_quick_replies = available_brands

            if not brand_quick_replies:
                brand_quick_replies = ["Any brand"]

    # ── Checkout info collection ───────────────────────────────────
    recipient = state.get("recipient", {})
    sender = state.get("sender", {})
    delivery_info = state.get("delivery_info", {})

    checkout_fields = {"city", "delivery_date", "recipient_name",
                       "recipient_phone", "sender_name", "sender_phone"}

    if state.get("intent") == "create_order" and checkout_fields & set(missing):
        extract_res = llm.invoke([
            SystemMessage(content=KAPRU_SYSTEM_PROMPT),
            HumanMessage(content=f"""Conversation so far: {state['messages'][-6:]}
Latest user message: "{last_msg}"

Extract any checkout details mentioned across the conversation. Return JSON:
{{
  "recipient_name": "" ,
  "recipient_phone": "",
  "sender_name": "",
  "sender_phone": "",
  "city": "",
  "delivery_date": "",
  "address": ""
}}
Rules:
- "name, phone" format like "sarath, 0771212121" -> fill name + phone
- A line with a city + date like "Nugegoda, 2026-06-30" -> city + delivery_date
- If sender said "same as sender" copy sender to recipient
- Leave a field "" if not mentioned. Return ONLY the fields you are confident about.""")
        ])
        try:
            info = parse_llm_json(extract_res.content)
            if info.get("recipient_name"):
                recipient["name"] = info["recipient_name"]
            if info.get("recipient_phone"):
                recipient["phone"] = info["recipient_phone"]
            if info.get("sender_name"):
                sender["name"] = info["sender_name"]
            if info.get("sender_phone"):
                sender["phone"] = info["sender_phone"]
            if info.get("city"):
                delivery_info["city"] = info["city"]
            if info.get("delivery_date"):
                delivery_info["delivery_date"] = info["delivery_date"]
            if info.get("address"):
                delivery_info["address"] = info["address"]

            # Recompute what's still missing
            if recipient.get("name"): missing = [m for m in missing if m != "recipient_name"]
            if recipient.get("phone"): missing = [m for m in missing if m != "recipient_phone"]
            if sender.get("name"): missing = [m for m in missing if m != "sender_name"]
            if sender.get("phone"): missing = [m for m in missing if m != "sender_phone"]
            if delivery_info.get("city"): missing = [m for m in missing if m != "city"]
            if delivery_info.get("delivery_date"): missing = [m for m in missing if m != "delivery_date"]
        except Exception as e:
            print(f"[CHECKOUT PARSE ERROR] {e}")

    # ── Step 4: Main reasoning decision ────────────────────────────
    prompt = f"""You are deciding the next action for Kapru,
a warm Sri Lankan shopping assistant.

Current plan step: "{current_plan_step}"
Full plan: {state['plan']}
Step index: {state['current_step']}
Language: {state['language']}
Cart: {state['cart']}
Missing info: {missing}
Price range collected: {price_range}
Price collected: {price_collected}
Selected brand: {selected_brand}
Available brands: {available_brands}
Cached products count: {len(cached_products)}
Delivery info: {state['delivery_info']}
Recipient: {state['recipient']}
Last tool result: {str(state.get('last_tool_result', ''))[:300]}
Last 3 messages: {state['messages'][-3:]}

Decide next action. Return JSON:
{{
    "action": "call_tool | clarify | respond",
    "tool": "kapruka_search_products | kapruka_get_product | kapruka_list_categories | kapruka_list_delivery_cities | kapruka_check_delivery | kapruka_create_order | kapruka_track_order | null",
    "tool_params": {{}},
    "clarify_question": "warm question in user language ({state['language']})",
    "quick_replies": [],
    "reasoning": "brief reason"
}}

Rules:
- "price_range" in missing -> action: clarify, quick_replies: {price_quick_replies}
- "brand" in missing AND price collected -> action: clarify, quick_replies: {brand_quick_replies}
- missing is empty AND cached products exist -> action: respond (use cache, do NOT call a tool)
- missing is empty AND no cache -> action: call_tool
  tool kapruka_search_products with tool_params q, min_price {price_range.get('min_price')}, max_price {price_range.get('max_price')}
- create_order step -> action: call_tool, tool kapruka_create_order
- check_delivery step -> action: call_tool, tool kapruka_check_delivery
- Plan complete -> action: respond

Personality: warm Sri Lankan friend, use Aney/Aiyo naturally."""

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    decision = parse_llm_json(res.content)

    print(f"🧠 [REASONER] action={decision.get('action')} tool={decision.get('tool')} price={price_range}")

    return {
        **state,
        "_decision": decision,
        "price_range": price_range,
        "cached_products": cached_products,
        "available_brands": available_brands,
        "selected_brand": selected_brand,
        "missing_info": missing,
        "recipient": recipient,      # ← add
        "sender": sender,            # ← add
        "delivery_info": delivery_info,  # ← add
    }


# ── TOOL CALLER NODE ─────────────────────────────────────────────────────────

async def tool_caller_node(state: ShoppingState) -> ShoppingState:
    """Executes the MCP tool decided by reasoner."""
    decision = state.get("_decision", {})
    print(f"\n⚡ [TOOL] calling {decision.get('tool')} params={decision.get('tool_params')}")

    tool_name = decision.get("tool")
    params = decision.get("tool_params", {})

    result = None
    new_delivery = state.get("delivery_info", {})

    try:
        if tool_name == "kapruka_search_products":
            result = await tools.search_products(
                q=params.get("q", ""),
                category=params.get("category"),
                min_price=params.get("min_price"),
                max_price=params.get("max_price"),
                in_stock_only=params.get("in_stock_only", True),
                limit=params.get("limit", 6),
                currency=params.get("currency", "LKR"),
            )

        elif tool_name == "kapruka_get_product":
            result = await tools.get_product(
                product_id=params.get("product_id"),
                currency=params.get("currency", "LKR"),
            )

        elif tool_name == "kapruka_list_categories":
            result = await tools.list_categories(depth=params.get("depth", 1))

        elif tool_name == "kapruka_list_delivery_cities":
            result = await tools.list_delivery_cities(
                query=params.get("query", ""),
                limit=params.get("limit", 10),
            )

        elif tool_name == "kapruka_check_delivery":
            result = await tools.check_delivery(
                city=params.get("city", ""),
                delivery_date=params.get("delivery_date", ""),
                product_id=params.get("product_id", ""),
            )
            new_delivery = {
                "city": params.get("city"),
                "delivery_date": params.get("delivery_date"),
            }

        elif tool_name == "kapruka_create_order":
            # Build the order from STATE, not from LLM params — deterministic + correct schema
            cart_items = []
            for item in state.get("cart", []):
                cart_items.append({
                    "product_id": item.get("product_id") or item.get("id"),
                    "quantity": item.get("qty", item.get("quantity", 1)),
                })

            recipient = state.get("recipient", {})
            sender = state.get("sender", {})
            delivery_info = state.get("delivery_info", {})

            delivery = {
                "address": delivery_info.get("address", delivery_info.get("city", "")),
                "city": delivery_info.get("city", ""),
                "date": delivery_info.get("delivery_date", ""),
            }

            result = await tools.create_order(
                cart=cart_items,
                recipient={"name": recipient.get("name", ""), "phone": recipient.get("phone", "")},
                delivery=delivery,
                sender={"name": sender.get("name", "")},
                gift_message=state.get("gift_message", ""),
                currency="LKR",
            )

        elif tool_name == "kapruka_track_order":
            result = await tools.track_order(
                order_number=params.get("order_number", ""),
            )

    except Exception as e:
        result = {"error": str(e)}

    print(f"⚡ [TOOL] result: {str(result)[:120]}")

    # Count consecutive errors to break replan loops
    retry = state.get("retry_count", 0)
    if isinstance(result, dict) and result.get("error"):
        retry += 1
    elif isinstance(result, str) and "Error" in result[:30]:
        retry += 1
    else:
        retry = 0

    return {
        **state,
        "last_tool_result": result,
        "delivery_info": new_delivery,
        "current_step": state["current_step"] + 1,
        "retry_count": retry,
    }


# ── REFLECTOR NODE ───────────────────────────────────────────────────────────

async def reflector_node(state: ShoppingState) -> ShoppingState:
    """Checks if last tool result was useful. Decides continue / replan / respond."""
    result_str = str(state.get("last_tool_result", ""))[:500]

    prompt = f"""Tool was called. Evaluate the result.

Tool result (first 500 chars): {result_str}
Current step: {state['current_step']}
Plan length: {len(state['plan'])}

Decide next move. Return JSON:
{{"status": "respond", "reason": "brief reason"}}

IMPORTANT RULES:
- If the tool returned products or any usable data -> status MUST be "respond"
- Only use "replan" if the result is clearly an error or completely empty
- NEVER use "continue" for a search result — always "respond" so the user sees results
- Default to "respond"."""

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    reflection = parse_llm_json(res.content)

    print(f"\n👁️  [REFLECTOR] status={reflection.get('status')}")

    return {
        **state,
        "reflection": reflection.get("status", "respond"),
    }


# ── RESPONDER NODE ───────────────────────────────────────────────────────────

async def responder_node(state: ShoppingState) -> ShoppingState:
    """Builds the final structured JSON response for the frontend."""
    decision = state.get("_decision", {})

    # Prefer cached products for product_cards when available
    products_context = state.get("cached_products", [])[:6]

    prompt = f"""Build a frontend response for Kapru.

Language: {state['language']}
Action decided: {decision.get('action')}
Clarify question: {decision.get('clarify_question', '')}
Quick replies: {decision.get('quick_replies', [])}
Products to show (from cache): {products_context}
Last tool result: {str(state.get('last_tool_result', ''))[:600]}
Cart: {state['cart']}
Last 3 messages: {state['messages'][-3:]}

Return ONE of these JSON response types:

product_cards (when showing search results — use the cached products above):
{{"type":"product_cards","message":"...","products":[{{"id":"...","name":"...","price":0,"image":"...","rating":4.5,"badge":"..."}}],"quick_replies":[]}}

question (when asking user for info):
{{"type":"question","message":"...","quick_replies":["option1","option2"]}}

delivery_quote (when showing check_delivery result):
{{"type":"delivery_quote","message":"...","quote":{{"city":"...","delivery_date":"...","fee":0,"arrives":"..."}}}}

cart_summary (when showing cart):
{{"type":"cart_summary","message":"...","items":[],"subtotal":0,"delivery_fee":0,"total":0}}

order_confirmation (when order is created):
{{"type":"order_confirmation","message":"...","order_id":"...","pay_link":"...","total":0,"delivery_time":"..."}}

text (for general messages):
{{"type":"text","message":"..."}}

When building product_cards, map each cached product: id=product_id, name=name, price=price.
Respond in the user's language ({state['language']}).
Return only valid JSON. No markdown."""

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    response = parse_llm_json(res.content)
    print(f"\n📤 [RESPONDER] type={response.get('type')}")

    return {
        **state,
        "response": response,
    }