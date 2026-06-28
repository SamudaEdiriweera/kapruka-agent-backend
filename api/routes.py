"""
What this file does

This is the HTTP layer — the single endpoint your Next.js frontend calls. 
It receives the chat request, sets up the initial state, runs the LangGraph agent, 
and returns the structured JSON response.

Why we need it

The graph knows nothing about HTTP. 
This bridges the frontend and the agent.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent.graph import graph
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

# ----- REQUEST SCHEMA ---------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    cart: Optional[List[dict]] = []
    language: Optional[str] = "en"
    delivery_info: Optional[dict] = {}
    recipient: Optional[dict] = {}
    sender: Optional[dict] = {}
    gift_message: Optional[str] = ""
    price_range: Optional[dict] = {}
    selected_brand: Optional[str] = ""
    available_brands: Optional[List[str]] = []
    cached_products: Optional[List[dict]] = []
    missing_info: Optional[List[str]] = []   # ← add this
    intent: Optional[str] = ""               # ← add this
    form_data: Optional[dict] = {}

# -----------------------------------------------------------

def build_dynamic_label(node: str, state: dict) -> str:
    """Build a context-aware status label from the node's real state."""
    intent = state.get("intent", "")
    cached = state.get("cached_products", [])
    delivery = state.get("delivery_info", {})
    missing = state.get("missing_info", [])
    decision = state.get("_decision", {})

    if node == "detector":
        intent = state.get("intent", "")
        if intent == "get_advice":
            return "Reading the situation…"
        if intent == "create_order":
            return "Getting ready to check out…"
        if intent == "track_order":
            return "Looking up your order…"
        if intent == "search_product":
            return "Understanding what you're looking for…"
        return "Understanding what you need…"

    if node == "planner":
        if intent == "create_order":
            return "Setting up your order…"
        if intent == "track_order":
            return "Looking up your order…"
        if intent == "get_advice":
            return "Thinking about what would help…"   # ← add
        return "Figuring out the best way to help…"

    if node == "reasoner":
        reason = state.get("_decision", {}).get("reasoning", "")
        if reason:
            return reason[:60]  # show the agent's actual reasoning

    if node == "tool_caller":
        tool = decision.get("tool", "")
        params = decision.get("tool_params", {})
        if tool == "kapruka_search_products":
            q = params.get("q", "products")
            return f"Searching Kapruka for {q}…"
        if tool == "kapruka_create_order":
            return "Placing your order on Kapruka…"
        if tool == "kapruka_check_delivery":
            city = delivery.get("city", "your area")
            return f"Checking delivery to {city}…"
        if tool == "kapruka_list_categories":
            return "Browsing Kapruka categories…"
        if tool == "kapruka_track_order":
            return "Tracking your order…"
        return "Fetching from Kapruka…"

    if node == "reflector":
        if cached:
            return f"Found {len(cached)} options, reviewing…"
        return "Checking the results…"

    if node == "responder":
        return "Putting it together…"

    return ""  # unknown node → skip


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # ── Merge structured form data (same as /chat) ──
    form = req.form_data or {}
    recipient = req.recipient or {}
    sender = req.sender or {}
    delivery = req.delivery_info or {}
    if form.get("recipient_name"):  recipient["name"] = form["recipient_name"]
    if form.get("recipient_phone"): recipient["phone"] = form["recipient_phone"]
    if form.get("sender_name"):     sender["name"] = form["sender_name"]
    if form.get("address"):         delivery["address"] = form["address"]
    if form.get("city"):            delivery["city"] = form["city"]
    if form.get("delivery_date"):   delivery["delivery_date"] = form["delivery_date"]

    initial_state = {
        "messages": [m.dict() for m in req.messages],
        "cart": req.cart,
        "plan": [],
        "current_step": 0,
        "language": req.language,
        "intent": req.intent,
        "missing_info": req.missing_info,
        "price_range": req.price_range,
        "cached_products": req.cached_products,
        "selected_brand": req.selected_brand,
        "available_brands": req.available_brands,
        "delivery_info": delivery,
        "recipient": recipient,
        "sender": sender,
        "gift_message": req.gift_message,
        "_decision": {},
        "reflection": "",
        "last_tool_result": None,
        "retry_count": 0,
        "response": None,
    }

    async def event_generator():
        final_state = None
        try:
            accumulated = dict(initial_state)
            async for chunk in graph.astream(initial_state):
                for node_name, node_state in chunk.items():
                    if isinstance(node_state, dict):
                        accumulated.update(node_state)   # merge partial updates
                        label = build_dynamic_label(node_name, accumulated)
                        if label:
                            yield f"data: {json.dumps({'type':'status','node':node_name,'label':label})}\n\n"
            final_state = accumulated

            # ── Final response ──
            if final_state and final_state.get("response"):
                final = {
                    "type": "final",
                    "response": final_state["response"],
                    "state": {
                        "cart": final_state.get("cart", []),
                        "language": final_state.get("language", "en"),
                        "price_range": final_state.get("price_range", {}),
                        "selected_brand": final_state.get("selected_brand", ""),
                        "available_brands": final_state.get("available_brands", []),
                        "cached_products": final_state.get("cached_products", []),
                        "delivery_info": final_state.get("delivery_info", {}),
                        "recipient": final_state.get("recipient", {}),
                        "sender": final_state.get("sender", {}),
                        "gift_message": final_state.get("gift_message", ""),
                        "missing_info": final_state.get("missing_info", []),
                        "intent": final_state.get("intent", ""),
                    },
                }
                yield f"data: {json.dumps(final)}\n\n"
            else:
                yield f"data: {json.dumps({'type':'error','message':'No response generated'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# -------- CHAT ENDPOINT ----------

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        # ── Merge structured form data into state (deterministic) ──
        form = req.form_data or {}
        recipient = req.recipient or {}
        sender = req.sender or {}
        delivery = req.delivery_info or {}

        if form.get("recipient_name"):  recipient["name"] = form["recipient_name"]
        if form.get("recipient_phone"): recipient["phone"] = form["recipient_phone"]
        if form.get("sender_name"):     sender["name"] = form["sender_name"]
        if form.get("address"):         delivery["address"] = form["address"]
        if form.get("city"):            delivery["city"] = form["city"]
        if form.get("delivery_date"):   delivery["delivery_date"] = form["delivery_date"]

        initial_state = {
            "messages": [m.dict() for m in req.messages],
            "cart": req.cart,
            "plan": [],
            "current_step": 0,
            "language": req.language,
            "intent": "",
            "missing_info": [],
            "price_range": req.price_range,
            "cached_products": req.cached_products,
            "selected_brand": req.selected_brand,
            "available_brands": req.available_brands,
            "delivery_info": req.delivery_info,
            "recipient": req.recipient,
            "sender": req.sender,
            "gift_message": req.gift_message,
            "missing_info": req.missing_info,   # ← add this
            "intent": req.intent      ,          # ← add this
            "_decision": {},
            "reflection": "",
            "last_tool_result": None,
            "response": None
        }

        result = await graph.ainvoke(initial_state)

        if not result.get("response"):
            raise HTTPException(status_code=500, detail="Agent returned no response")
        
        return {
            "response": result["response"],
            "state": {
                "cart": result["cart"],
                "language": result["language"],
                "price_range": result["price_range"],
                "selected_brand": result["selected_brand"],
                "available_brands": result["available_brands"],
                "cached_products": result["cached_products"],
                "delivery_info": result["delivery_info"],
                "recipient": result["recipient"],
                "sender": result["sender"],
                "gift_message": result["gift_message"],
                "missing_info": result["missing_info"],  # ← add this
                "intent": result["intent"]               # ← add this
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# -------- HEALTH -----------------------------

@router.get("/health")
def health():
    return {"status": "ok", "agent": "kapru"}