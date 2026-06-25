"""
Web Search Tool — DuckDuckGo-powered search for location and general queries.
Integrates MCP (Model Context Protocol) web search capabilities into RAG pipeline.
"""

import httpx
from typing import Optional
from bs4 import BeautifulSoup


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return top results.
    
    Args:
        query: The search query (e.g., "Paris tourist attractions")
        max_results: Number of results to return (default 5, max 10)
    
    Returns:
        Formatted search results with title, URL, and snippet.
    """
    max_results = min(max_results, 10)
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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
    for i, result in enumerate(soup.select(".result"), 1):
        if i > max_results:
            break
        
        title_elem = result.select_one(".result__title a")
        snippet_elem = result.select_one(".result__snippet")
        
        if not title_elem:
            continue
        
        title = title_elem.get_text(strip=True)
        link = title_elem.get("href", "")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description available"
        
        results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")

    if not results:
        return f"No web search results found for: '{query}'"

    return f"Web Search Results for '{query}':\n\n" + "\n\n".join(results)


def search_location_info(location_name: str) -> str:
    """
    Search for detailed information about a specific location.
    Useful when user asks about places mentioned in documents.
    
    Args:
        location_name: Name of the location (e.g., "Paris", "London", "Mumbai")
    
    Returns:
        Location details from web search.
    """
    query = f"{location_name} information tourist attractions climate weather"
    return web_search(query, max_results=3)


def search_general_info(topic: str) -> str:
    """
    General purpose web search for any topic not found in documents.
    
    Args:
        topic: Topic to search
    
    Returns:
        Web search results.
    """
    return web_search(topic, max_results=5)
