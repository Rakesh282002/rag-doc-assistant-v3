"""
MCP Server — Exposes location search and maps tools via Model Context Protocol.

Run standalone:
    python -m backend.mcp_server

Or used as a subprocess by the MCP client in the RAG pipeline.
"""

import urllib.parse

import httpx
from bs4 import BeautifulSoup
from mcp.server import FastMCP

# Create MCP server instance
mcp = FastMCP(
    name="rag-location-tools",
    instructions="Provides location search and Google Maps link generation tools for the RAG Document Assistant.",
)


@mcp.tool(
    name="web_search",
    description="Search the web using DuckDuckGo. Returns top results with title, URL, and snippet.",
)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results."""
    max_results = min(max_results, 10)
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = httpx.post(url, data={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error performing web search: {str(e)}"

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        return f"Error parsing search results: {str(e)}"

    results = []
    for i, result_elem in enumerate(soup.select(".result"), 1):
        if i > max_results:
            break

        title_elem = result_elem.select_one(".result__title a")
        snippet_elem = result_elem.select_one(".result__snippet")

        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        link = title_elem.get("href", "")
        snippet = (
            snippet_elem.get_text(strip=True)
            if snippet_elem
            else "No description available"
        )

        results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")

    if not results:
        return f"No web search results found for: '{query}'"

    return f"Web Search Results for '{query}':\n\n" + "\n\n".join(results)


@mcp.tool(
    name="search_location_info",
    description="Search for detailed information about a specific location including tourist attractions, climate, and weather.",
)
def search_location_info(location_name: str) -> str:
    """Search web for location-specific information."""
    query = f"{location_name} information tourist attractions climate weather"
    return web_search(query, max_results=3)


@mcp.tool(
    name="generate_maps_link",
    description="Generate a Google Maps search URL for a given location. Returns a markdown-formatted clickable link.",
)
def generate_maps_link(location: str) -> str:
    """Generate Google Maps link for the given location."""
    encoded = urllib.parse.quote_plus(location)
    url = f"https://www.google.com/maps/search/{encoded}"
    return f"[\U0001f4cd View on Google Maps]({url})"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Location Tools Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (internal/subprocess) or sse (persistent HTTP service)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE mode")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"[MCP SERVER] Starting SSE server on http://{args.host}:{args.port}/sse")
        print(f"[MCP SERVER] Tools: web_search, search_location_info, generate_maps_link")
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
