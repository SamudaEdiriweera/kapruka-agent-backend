"""
What this file does?
This is the brain of the entire agent. 
Five nodes, each with a specific job:

    DETECTOR  → reads user message, detects language + intent
    PLANNER   → builds a step-by-step shopping plan
    REASONER  → decides what to do right now
    TOOL CALLER → executes the right Kapruka MCP tool
    REFLECTOR → checks if tool result was good, decides next move
    RESPONDER → builds structured JSON for the frontend
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ShoppingState
from agent import tools

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

KAPRU_SYSTEM_PROMPT = """You are Kapru, a warm and clever Sri Lankan shopping assistant for kapruka.
You speak English, Sinhala(සිංහල), and Tanglish naturally.
Always match the language the user is writing in.
Be warm, helpful and local - like a smart friend, not a corporate bot.
Use natural Sri Lankan expressions when appropriate (Aney, Aiyo, Machan etc).
ALWAYS respond with valid JSON only. No markdown. No extra text. No Code fences. Just JSON.

"""

# ----- helpers ---------------------------------------------------

def parse_llm_json(content: str) -> dict:
    """ Safetly parse LLM JSON response, strip fences if present."""
    clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)

def get_last_user_message(state: ShoppingState) -> str:
    """ Get the most recent user message from conversation history."""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""

# -------------- DETECTOR NODE --------------------------------

async def detector_node(state: ShoppingState) -> ShoppingState:
    """
    Reads the latest user message.
    Detects: language, intent, subject, missing info.
    """
    last_msg = get_last_user_message(state)

    prompt = f"""Analyze this message from a Kapruka shopping user: "{last_msg}"

    Return JSON:
    {{
        "language": "en" | "si" | "tl",
        "intent": "search_product | get_product | list_categories | check_delivery | create_order | track_order | add_to_cart | view_cart | general",
        "subject": "external product name or topic or empty string",
        "missing_info": ["city", "delivery_date", "recipient_name", "recipient_phone", "sender_name", "sender_phone"]
    }}

    missing_info should only include fields actually needed for the detectedd intent that are not in the conversation yet.

"""
    
    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    parsed = parse_llm_json(res.content)

    return {
        **state,
        "language": parsed.get("language", state.get("language", "en")),
        "intent": parsed.get("intent", "general"),
        "missing_info": parsed.get("missing_info", [])
    }

# -------------- PLANNER NODE --------------------------------

async def planner_node(state: ShoppingState) -> ShoppingState:
    """
    Builds a full step-by-step plan based on intent.
    Only replans if plan is empty or reflection triggered a replan.
    """
    if state.get("plan") and state.get("reflection") != "replan":
        return state
    
    prompt = f"""User intent: "{state['intent']}"
Language: "{state['language']}"
Cart: {state['cart']}
Missing info: {state['missing_info']}
Conversation: {state['messages'][-4:]}

Buils a minimal shopping plan as a JSON array  of action strings.
Available actions: search_product, get_product, list_categories, check_delivery,
create_order, track_order, ask_missing_info, show_cart, respond

Example for "send birthday cake":
["search_product", "show_cards", "ask_missing_info", "check_delivery", "ask_gift_message", "create_order", "show_confirmation"]

Return only the JSON array.

"""
    
    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    plan = parse_llm_json(res.content)

    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "reflection": ""
    }

# --------------------- REASONER NODE ---------------------------

async def reasoner_node(state: ShoppingState) -> ShoppingState:
    """ 
    Looks at current plan stop + state.
    Decides: call a tool, ask user for missing info, or respond.
    """
    current_plan_step = state["plan"][state["current_step"]] if state["plan"] else "respond"

    prompt = f"""You are deciding the next action for kapru.

    Current plan step: "{current_plan_step}"
    Full plan: {state['plan']}
    Step index: {state['current_step']}
    Language: {state['language']}
    Cart: {state['cart']}
    Missing info: {state['missing_info']}
    Delivery info: {state['delivery_info']}
    Recipient: {state['recipient']}
    Last tool result summary: {str(state.get('last_tool_result', ''))[:300]}
    Last 3 messages: {state['messages'][-3:]}

    Decide the next action. Return JSON:
    {{
    "action": "call_tool | clarify | respond",
    "tool": "kapruka_search_products | kapruka_get_product | kapruka_list_categories | kapruka_list_delivery_cities | kapruka_check_delivery | kapruka_create_order | kapruka_track_order | null",
    "tool_params": {{}},
    "clarify_question": "question to ask user if action is clarify, in the user's language"
    "quick_replies": ["option1", "option2"],
    "reasoning": "brief reason"
    }}

    Rules:
        - If missing_info is not empty and needed for next step -> action: clarify
        - If tool_params are all available -> action: call_tool
        - If plan is complete or no tool needed -> action: respon"""
    
    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    decision = parse_llm_json(res.content)

    return {
        **state,
        "_decision": decision

    }

# --------------------- TOOL CALLER ---------------------------

async def tool_caller_node(state: ShoppingState) -> ShoppingState:
    """
    Executes the MCP tool decided by reasoner.
    Updates cart if product was added.
    Updates delivery_info / recipient from tool params.
    """
    decision = state.get("_decision", {})
    tool_name = decision.get("tool")
    params = decision.get("tool_params", {})

    result = None

    try:
        if tool_name == "kapruka_search_products":
            result = await tools.search_products(
                q=params.get("q", ""),
                category=params.get("category"),
                min_price=params.get("min_price"),
                max_price=params.get("max_price"),
                in_stock_only=params.get("in_stock_only", True),
                limit=params.get("limit", 6),
                currency=params.get("currency", "LKR")
            )
        
        elif tool_name == "kapruka_get_product":
            result = await tools.get_product(
                product_id=params.get("product_id"),
                currency=params.get("currency", "LKR")
            )

        elif tool_name == "kapruka_list_categories":
            result = await tools.list_categories(
                depth=params.get("depth", 1)
            )

        elif tool_name == "kapruka_list_delivery_cities":
            result = await tools.list_delivery_cities(
                query=params.get("query", ""),
                limi=params.get("limit", 10)
            )

        elif tool_name == "kapruka_check_delivery":
            result = await tools.check_delivery(
                city=params.get("city", ""),
                delivery_date=params.get("delivery_date", ""),
                product_id=params.get("product_id", "")
            )
            # persist delivery info
            state["delivery_info"] = {
                "city": params.get("city"),
                "delivery_date": params.get("delivery_date")
            }

        elif tool_name == "kapruka_create_order":
            result = await tools.create_order(
                cart=state["cart"],
                recipient=state["recipient"],
                delivery=state["delivery_info"],
                sender=state["sender"],
                gift_message=state.get("gift_message", ""),
                currency=params.get("currency", "LKR")
            )

        elif tool_name == "kapruka_track_order":
            result = await tools.track_order(
                order_number=params.get("order_number", "")
            )

    except Exception as e:
        result = {"error": str(e)}

    return {
        **state,
        "last_tool_result": result,
        "current_step": state["current_step"] + 1
    }

# --------------------- REFLECTOR NODE ---------------------------

async def reflector_node(state: ShoppingState) -> ShoppingState:
    """
    Checks if last tool result was useful.
    Decides: continue to next step, replan, or respond to user now. 
    """
    result_str = str(state.get("last_tool_result", ""))[:500]

    prompt = f"""Tool was called. Evaluate the result.

    Tool result: {result_str}
    Current step: {state['current_step']}
    Plan: {state['plan']}
    Cart: {state['cart']}

    Decide next move. Return JSON:
    {{
        "status": "continue | replan | respond",
        "reason": "brief reason"
    }}

    Rules:
    - "contine" -> result was good, move to next plan step
    - "replan" -> result was empty/error, need a new plan
    - "repond" -> we have enough to respond to user now
    """

    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    reflection = parse_llm_json(res.content)

    return {
        **state,
        "reflection": reflection.get("status", "respond")
    }

# --------------------- RESPONDER NODE ---------------------------

async def responder_node(state: ShoppingState) -> ShoppingState:
    """
    Builds the final strcutured JSON response for the frontend.
    Matches one of the response types the frontend renders.
    """

    decision = state.get("_decision", {})

    prompt = f"""Build a frontend response for Kapru.

    Language: {state['language']}
    Action decided: {decision.get('action')}
    Clarify question: {decision.get('clarify_question', '')}
    Quick replies: {decision.get('quick_replies', [])}
    Last tool result: {str(state.get('last_tool_result', ''))[:600]}
    Cart: {state['cart']}
    Last 3 messages: {state['messages'][-3:]}

    Return ONE of these JSON response types:

    product_cards (when showing search results):
    {{"type":"product_cards","message":"...","products":[{{"id":"...","name":"...","price":0, "image":"...","rating":0.0,"badge":"..."}}],"quick_replies":[]}}

    question (when asking user for info):
    {{"type":"question","message":"...","quick_replies":["option1","option2"]}}

    delivery (when showing check_delivery result):
    {{"type":"delivery_quote","message":"...","quote":{{"city":"...","delivery_date":"...","fee":0,"arrives":"..."}}}}

    cart_summary (when showing cart):
    {{"type":"cart_summary","message":"...","items":[],"subtotal":0,"delivery_fee":0,"total":0}}

    order_confirmation (when order is created ):
    {{"type":"order_confirmation","message":"...","order_id":"...","pay_link":"...","total":0,"delivery_time":"..."}}

    text (for general messages)
    {{"type":"text","message":"..."}}

    Respond in the user's language ({state['language']}).
    Retuen only valid JSON. No markdown.
"""
    
    res = llm.invoke([
        SystemMessage(content=KAPRU_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    response = parse_llm_json(res.content)

    return {
        **state,
        "response": response
    }
