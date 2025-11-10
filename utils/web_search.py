"""
utils/web_search.py
Provides real-time web search using SerpAPI.
"""

import os
import requests
from config.config import SERPAPI_API_KEY


def web_search(query, num_results=3):
    """
    Perform a live Google search using SerpAPI.
    Returns a formatted text summary of results.
    """
    if not SERPAPI_API_KEY:
        return "⚠️ Web search unavailable (missing SerpAPI key)."

    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "num": num_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  # ensures HTTP errors are caught
        data = response.json()

        results = []
        for result in data.get("organic_results", [])[:num_results]:
            title = result.get("title", "No title")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            results.append(f"🔹 **{title}**\n{snippet}\n{link}")

        if not results:
            return "⚠️ No relevant results found online."

        return "\n\n".join(results)

    except Exception as e:
        return f"⚠️ Web search failed: {str(e)}"
