"""
What this file does

This is the bridge between your agent and Kapruka's live platform.
Every time the agent needs real data - products, delivery quotes, orders -
it calls through here. Nothing else in your codebase talks to Kapruka directly.

The Kapruka MCP uses streamable HTTP transport (requires session handshake)
and returns results as MARKDOWN TEXT, not JSON. So this file:
  1. Uses the official MCP SDK to handle the session handshake automatically
  2. Parses the markdown product list into structured dicts
"""

import os
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

MCP_URL = os.getenv("KAPRUKA_MCP_URL", "https://mcp.kapruka.com/mcp")


# ── MCP SESSION ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def mcp_session():
    """Open an MCP session with proper handshake + session handling."""
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_mcp(tool_name: str, params: dict) -> str:
    """
    Call any Kapruka MCP tool. Returns the raw markdown text result.
    NOTE: Kapruka nests arguments under a 'params' key.
    """
    try:
        async with mcp_session() as session:
            result = await session.call_tool(tool_name, {"params": params})
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts)
    except Exception as e:
        # Surface the real error instead of a vague TaskGroup wrapper
        print(f"[MCP ERROR] {tool_name}: {type(e).__name__}: {e}")
        return f"Error: MCP call failed - {str(e)[:100]}"


# ── MARKDOWN PARSER ──────────────────────────────────────────────────────────────

def parse_products(markdown: str) -> list:
    """
    Parse Kapruka's markdown product list into structured dicts.

    Expected format per product:
      **1. Product Name**
         ID: `PRODUCT_ID` · LKR 4,290 · In stock (low) · ships internationally
         [View product](https://...)
    """
    products = []

    # Each product block starts with **N. Name**
    # Split on the numbered bold headers
    blocks = re.split(r"\*\*\d+\.\s", markdown)

    for block in blocks[1:]:  # skip header text before first product
        try:
            # Name is everything up to the closing **
            name_match = re.match(r"(.+?)\*\*", block, re.DOTALL)
            name = name_match.group(1).strip() if name_match else ""

            # Product ID inside backticks
            id_match = re.search(r"ID:\s*`([^`]+)`", block)
            product_id = id_match.group(1).strip() if id_match else ""

            # Price after "LKR"
            price_match = re.search(r"LKR\s*([\d,]+)", block)
            price = int(price_match.group(1).replace(",", "")) if price_match else 0

            # Product URL
            url_match = re.search(r"\[View product\]\((https?://[^\)]+)\)", block)
            url = url_match.group(1).strip() if url_match else ""

            # Stock status
            in_stock = "in stock" in block.lower()

            if name and product_id:
                products.append({
                    "product_id": product_id,
                    "name": name,
                    "price": price,
                    "url": url,
                    "in_stock": in_stock,
                    "image": ""  # MCP doesn't return images in search; frontend uses placeholder
                })
        except Exception:
            continue

    return products


# ── TOOL WRAPPERS ────────────────────────────────────────────────────────────────

async def search_products(
    q: str,
    category=None,
    min_price=None,
    max_price=None,
    in_stock_only: bool = True,
    sort: str = "relevance",
    limit: int = 6,
    currency: str = "LKR",
) -> dict:
    """
    Search Kapruka catalog. Returns {"products": [...], "raw": markdown}.
    """
    params = {
        "q": q,
        "in_stock_only": in_stock_only,
        "sort": sort,
        "limit": limit,
        "currency": currency,
    }
    if category:
        params["category"] = category
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price

    raw = await call_mcp("kapruka_search_products", params)
    return {"products": parse_products(raw), "raw": raw}


async def get_product(product_id: str, currency: str = "LKR") -> str:
    return await call_mcp("kapruka_get_product", {
        "product_id": product_id,
        "currency": currency,
    })


async def list_categories(depth: int = 1) -> str:
    return await call_mcp("kapruka_list_categories", {"depth": depth})


async def list_delivery_cities(query: str, limit: int = 10) -> str:
    return await call_mcp("kapruka_list_delivery_cities", {
        "query": query,
        "limit": limit,
    })


async def check_delivery(city: str, delivery_date: str, product_id: str) -> str:
    return await call_mcp("kapruka_check_delivery", {
        "city": city,
        "delivery_date": delivery_date,
        "product_id": product_id,
    })


async def create_order(cart, recipient, delivery, sender, gift_message="", currency="LKR") -> str:
    return await call_mcp("kapruka_create_order", {
        "cart": cart,
        "recipient": recipient,
        "delivery": delivery,
        "sender": sender,
        "gift_message": gift_message,
        "currency": currency,
    })


async def track_order(order_number: str) -> str:
    return await call_mcp("kapruka_track_order", {"order_number": order_number})


# ── HELPERS ──────────────────────────────────────────────────────────────────────

async def get_price_range_hints(q: str) -> dict:
    """
    Searches Kapruka for real prices and builds smart brackets from
    the actual price distribution. Caches products to avoid a 2nd MCP call.
    """
    try:
        result = await search_products(q=q, limit=12)
        products = result.get("products", [])
        prices = [p["price"] for p in products if p.get("price")]

        if not prices:
            return {"brackets": [], "products": []}

        sorted_prices = sorted(prices)
        n = len(sorted_prices)
        p25 = sorted_prices[n // 4]
        p50 = sorted_prices[n // 2]
        p75 = sorted_prices[(n * 3) // 4]

        brackets = [
            f"Under LKR {int(p25):,}",
            f"LKR {int(p25):,}–{int(p50):,}",
            f"LKR {int(p50):,}–{int(p75):,}",
            f"Above LKR {int(p75):,}",
        ]

        return {
            "brackets": brackets,
            "products": products,
            "min": min(prices),
            "max": max(prices),
        }
    except Exception:
        return {"brackets": [], "products": []}


def extract_brands(products: list) -> list:
    """
    Extract unique brand names from cached products.
    Uses the first word of each product name as the brand guess.
    """
    brands = set()
    for p in products:
        brand = p.get("brand") or p.get("name", "").split()[0]
        if brand and len(brand) > 1:
            brands.add(brand.strip())

    brands_list = sorted(list(brands))[:6]
    brands_list.append("Any brand")
    return brands_list