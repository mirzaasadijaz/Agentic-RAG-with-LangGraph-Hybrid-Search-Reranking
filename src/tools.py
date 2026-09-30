"""
Web search fallback, used when the vectorstore has nothing relevant for the
question (either the router sent us straight here, or grading + retries
came up empty). Requires a free Tavily API key: https://tavily.com
"""
import os

from . import config

_web_search_tool = None


def get_web_search_tool():
    global _web_search_tool
    if _web_search_tool is None:
        if not os.getenv("TAVILY_API_KEY"):
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
                "and add it to your .env file to enable the web_search fallback node."
            )
        from langchain_tavily import TavilySearch

        _web_search_tool = TavilySearch(max_results=config.WEB_SEARCH_K)
    return _web_search_tool
