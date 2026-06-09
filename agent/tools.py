"""
What this file does

This is the bridge between your agent and Kapruka's live platform. 
Every time the agent needs real data products, delivery quotes, orders 
it calls through here. Nothing else in your codebase talks to Kapruka directly.

Why we need it

Single responsibility — all MCP calls in one place
Easy to debug — if Kapruka returns bad data, you know exactly where to look
Clean nodes — agent nodes stay focused on logic, not HTTP calls
"""

import httpx
import os
from typing import Any, Optional

MCP_URL = os.getenv("KAPRUKA_MCP_URL", "https://mcp.kapruka.com/mcp")

HEADERS = {
    "content-type": "application/json",
    "Accept": "application/json, text/event-stream"
}

async def call_mcp(tool_name: str, params: dict) -> Any:
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            MCP_URL,
            headers=HEADERS,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 1,
                "params": {
                    "name": tool_name,
                    "arguments": params
                    }
            }
        )
        data = res.json()
        return data.get("result", {})
    
async def search_products(
        q: str,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock_only: bool = True,
        sort: str = "relevance",
        limit: int = 6,
        currency: str = "LKR"

):
    params = {
        "q": q,
        "in_stock_only": in_stock_only,
        "sort": sort,
        "limit": limit,
        "currency": currency
    }
    
    if category: params["category"] = category
    if min_price: params["min_price"] = min_price
    if max_price: params["max_price"] = max_price
    
    return await call_mcp(
        "kapruka_search_products", 
        params
        )

async def get_product(product_id: str, currency: str = "LKR") -> dict:
    return await call_mcp(
        "kapruka_get_product",
        {
        "product_id": product_id,
        "currency": currency
        } 
        )

async def list_categories(depth: int = 1):
    return await call_mcp(
        "kapruka_list_categories",
        {
            "depth": depth
        }
    )

async def list_delivery_cities(query: str, limit: int = 10):
    return await call_mcp(
        "kapruka_list_delivery_cities",
        {
            "query": query,
            "limit": limit
        }
    )

async def check_delivery(city_id: str, delivery_date: str, product_id: str):
    return await call_mcp(
        "kapruka_check_delivery",
        {
            "city_id": city_id,
            "delivery_date": delivery_date,
            "product_id": product_id
        }
    )

async def create_order(
        cart: list,
        recipient: dict,
        delivery: dict,
        sender: dict,
        gift_message: str = "",
        currency: str = "LKR"
):
    return await call_mcp(
        "kapruka_create_order",
        {
            "cart": cart,
            "recipient": recipient,
            "delivery": delivery,
            "sender": sender,
            "gift_message": gift_message,
            "currency": currency
        }
    )

async def track_order(order_number: str):
    return await call_mcp(
        "kapruka_track_order",
        {
            "order_number": order_number
        }
    )

async def get_price_range_hints(q: str) -> dict:
    """
    Quick search to get real price brackets from Kapruka's live catalog.
    Used to show accurate price range quick replies to user.
    """
    
    try: 
        result = await search_products(q=q, limit=6)
        products = result.get("products", [])
        prices = [p["price"] for p in products if "price" in p]

        if not prices:
            return {}
        
        min_p = min(prices)
        max_p = max(prices)
        mid = (min_p + max_p) / 2

        return {
            "min": min_p,
            "max": max_p,
            "brackets": [
                f"Under LKR {int(min_p):,}",
                f"LKR {int(min_p):,}-{int(mid):,}",
                f"LKR {int(mid):,}-{int(max_p):,}",
                f"Above LKR {int(max_p):,}"
            ]
        }
    except:
        return {}