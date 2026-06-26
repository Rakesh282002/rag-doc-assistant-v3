"""
MCP Client — Connects to the MCP server (mcp_server.py) via stdio or SSE transport
and exposes tool-calling functions for the RAG pipeline.

Transport selection:
    - Set MCP_SERVER_URL env var for SSE mode (e.g. "http://localhost:8000/sse")
    - Leave unset for stdio mode (auto-spawns server subprocess)

Usage:
    from backend.mcp_client import call_generate_maps_link, call_search_location_info
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


# Path to the MCP server script (for stdio mode)
_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Remote MCP server URL (set this to use persistent remote server)
# For SSE: http://host:port/sse
# For streamable-http: http://host:port/mcp
_MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "")


@asynccontextmanager
async def _get_session():
    """Get an MCP client session — auto-selects transport based on config."""
    if _MCP_SERVER_URL:
        if "/sse" in _MCP_SERVER_URL:
            # SSE mode
            async with sse_client(_MCP_SERVER_URL) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        else:
            # Streamable HTTP mode (default for remote)
            async with streamablehttp_client(_MCP_SERVER_URL) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
    else:
        # stdio mode: spawn server as subprocess
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


async def _call_tool(tool_name: str, arguments: dict) -> str:
    """
    Connect to MCP server, call a tool, and return the text result.

    Args:
        tool_name: Name of the MCP tool to call
        arguments: Dict of arguments to pass to the tool

    Returns:
        Tool result as string
    """
    async with _get_session() as session:
        result = await session.call_tool(tool_name, arguments)

        # Extract text content from result
        if result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return ""


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. Streamlit) — use a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    else:
        return asyncio.run(coro)


def call_generate_maps_link(location: str) -> str:
    """
    Call the MCP generate_maps_link tool.

    Args:
        location: Place name (e.g. "Bhimavaram, Andhra Pradesh")

    Returns:
        Markdown-formatted Google Maps link
    """
    return _run_async(_call_tool("generate_maps_link", {"location": location}))


def call_search_location_info(location_name: str) -> str:
    """
    Call the MCP search_location_info tool.

    Args:
        location_name: Name of the location to search

    Returns:
        Web search results about the location
    """
    return _run_async(_call_tool("search_location_info", {"location_name": location_name}))


def call_web_search(query: str, max_results: int = 5) -> str:
    """
    Call the MCP web_search tool.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        Formatted search results
    """
    return _run_async(
        _call_tool("web_search", {"query": query, "max_results": max_results})
    )


def list_available_tools() -> list:
    """List all tools exposed by the MCP server."""

    async def _list():
        async with _get_session() as session:
            result = await session.list_tools()
            return [
                {"name": t.name, "description": t.description}
                for t in result.tools
            ]

    return _run_async(_list())
