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


# -------- CHAT ENDPOINT ----------

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
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
            # Return updated state fields so frontend can persist them
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
                "gift_message": result["gift_message"]
            }

        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# -------- HEALTH -----------------------------

@router.get("/health")
def health():
    return {"status": "ok", "agent": "kapru"}