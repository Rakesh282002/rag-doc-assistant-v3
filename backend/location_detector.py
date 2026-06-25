"""
Location Detector — Identifies location-related queries and suggests web search enrichment.
Part of MCP integration for enhanced RAG capabilities.
"""

import re
import urllib.parse
from typing import Optional, Tuple


# Location-related keywords that trigger web search
LOCATION_KEYWORDS = {
    "location", "city", "town", "country", "region", "state", "province",
    "district", "area", "place", "where", "situated", "located", "geography",
    "address", "timezone", "coordinates", "latitude", "longitude",
    "climate", "weather", "attractions", "landmarks", "tourist", "visit",
    "distance", "nearby", "neighborhood", "headquarters", "office",
}

# Question patterns that indicate location interest
LOCATION_QUESTION_PATTERNS = [
    r"where\s+(?:is|are|was|were)",
    r"what\s+(?:is|are)\s+.*\s+location",
    r"(?:city|town|country|region)\s+(?:is|are)",
    r"(?:location|place|headquarters)\s+of",
    r"(?:located|situated)\s+(?:in|at|near)",
    r"(?:distance|near|distance\s+to)",
]

# Company/person identifiers to check for location context
ENTITY_INDICATORS = {
    "company", "firm", "organization", "org", "office", "workplace",
    "person", "candidate", "employee", "professional", "author",
}


def is_location_query(question: str) -> bool:
    """
    Detect if a question is asking about location details.
    
    Args:
        question: User question/query
    
    Returns:
        True if question appears to be location-related
    """
    q_lower = question.lower().strip()
    
    # Check for explicit location keywords
    if any(keyword in q_lower for keyword in LOCATION_KEYWORDS):
        return True
    
    # Check for question patterns
    for pattern in LOCATION_QUESTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    
    return False


def extract_location_from_query(question: str) -> Optional[str]:
    """
    Extract the location entity from a location-related question.
    
    Args:
        question: User question
    
    Returns:
        Extracted location name, or None if not found
    
    Examples:
        "Where is Microsoft located?" -> "Microsoft"
        "What is the climate in Paris?" -> "Paris"
    """
    q_lower = question.lower()
    
    # Pattern: "where is [entity] located"
    match = re.search(r"where\s+(?:is|are)\s+([^?]+?)(?:\s+located|$)", q_lower)
    if match:
        entity = match.group(1).strip()
        # Filter out common stop words
        entity = re.sub(r"^(?:the|a|an)\s+", "", entity)
        return entity if entity else None
    
    # Pattern: "what is the location of [entity]"
    match = re.search(r"(?:what\s+is\s+)?(?:the\s+)?location\s+of\s+([^?]+)$", q_lower)
    if match:
        return match.group(1).strip()
    
    # Pattern: "city/country of [entity]"
    match = re.search(r"(?:city|country|region|headquarters)\s+of\s+([^?]+)$", q_lower)
    if match:
        return match.group(1).strip()
    
    # Pattern: "[entity] is in/at [location]" (capturing location)
    match = re.search(r"(?:in|at|near)\s+([A-Za-z\s]+?)(?:\?|$)", q_lower)
    if match:
        location = match.group(1).strip()
        if location and len(location) < 50:
            return location
    
    return None


def should_augment_with_web_search(answer: str, question: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if an answer should be augmented with web search results.
    
    Args:
        answer: LLM-generated answer from RAG
        question: Original user question
    
    Returns:
        Tuple of (should_search: bool, search_query: Optional[str])
    """
    answer_lower = answer.lower()

    # Do not augment when the answer explicitly states the information
    # is not in the document.
    if "not mentioned" in answer_lower or "not found" in answer_lower:
        return False, None

    # Only augment for location questions when we can detect a concrete
    # location from the answer itself (document-grounded behavior).
    if is_location_query(question):
        location = extract_location_from_answer(answer)
        if location and len(answer.strip()) < 140:
            return True, location
    
    return False, None


def format_web_search_context(location: str) -> str:
    """
    Format web search query for location enrichment.
    
    Args:
        location: Location name to search
    
    Returns:
        Formatted search query
    """
    return f"{location} information details climate attractions"


# Patterns to detect "City, State" or "City, Country" style location mentions in answer text
_LOCATION_MENTION_PATTERNS = [
    # "Bhimavaram, Andhra Pradesh" or "Paris, France"
    r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b",
    # Standalone capitalized place names of 2+ words like "Andhra Pradesh"
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b",
]


def extract_location_from_answer(answer: str) -> Optional[str]:
    """
    Extract a location name (city/state/country) mentioned in the answer text.

    Args:
        answer: LLM-generated answer string

    Returns:
        Best location string found, or None
    """
    # Prefer "City, State" pattern first (most precise)
    match = re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b",
        answer,
    )
    if match:
        candidate = match.group(1).strip()
        # Reject overly long matches or common false positives
        if len(candidate) < 80 and candidate not in {"Not Mentioned", "Not Found"}:
            return candidate
    return None


def generate_maps_link(location: str) -> str:
    """
    Generate a Google Maps search URL for the given location.

    Args:
        location: Place name (e.g. "Bhimavaram, Andhra Pradesh")

    Returns:
        Markdown-formatted Google Maps link string
    """
    encoded = urllib.parse.quote_plus(location)
    url = f"https://www.google.com/maps/search/{encoded}"
    return f"[📍 View on Google Maps]({url})"
