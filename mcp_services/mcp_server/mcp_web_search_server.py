import os
from typing import Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests
from pydantic import BaseModel

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
load_dotenv()
mcp = FastMCP("web-search")
URL = "https://serpapi.com/search"


class SearchResult(BaseModel):
    """
    Represents the structured result of a web search operation.
    """
    tool: str
    query: str
    answer: list[Dict[str, Any]]
    sources: list[str]
    total_results: int


async def make_serpapi_request(query: str) -> dict:
    """
    Performs an HTTP GET request to the SerpAPI Google search API.

    This asynchronous function constructs the request parameters, including the
    search query, API key, desired engine, number of results, and safe search settings.
    It then executes the request and parses the JSON response.

    Args:
        query (str): The search query string.

    Returns:
        dict: Parsed JSON response from SerpAPI containing search results and metadata.

    Raises:
        Exception: If the HTTP response status code is not 200 or if an error
                   message is returned by the SerpAPI.
    """
    params = {
        'q': query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 5,
        "safe": "active",
    }

    response = requests.get(URL, params=params)
    data = response.json()
    if response.status_code != 200:
        raise Exception(f"Error fetching data from SerpAPI: {data.get('error', 'Unknown error')}")
    return data


def format_search_results(data: dict, query: str) -> str | SearchResult:
    """
    Converts the raw search results into a formatted summary string or a structured SearchResult model.

    Args:
        data (dict): The JSON response dictionary from SerpAPI API.
        query (str): The original search query string.

    Returns:
        str | SearchResult:
            - A SearchResult Pydantic object encapsulating the tool name, query, answer summary, and sources.
            - A fallback message if no results were returned by the API.
    """
    results = []
    for result in data.get("organic_results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", ""),
            "source": result.get("source", "")
        })
    return SearchResult(
        tool="WebSearchTool",
        query=query,
        answer=results,
        sources=[result.get("link", "") for result in data.get("organic_results", [])],
        total_results=len(results),
    )


class MCPWebSearchServer:
    @mcp.tool()
    async def run(self, refined_query: str) -> str | SearchResult:
        """
                MCP tool method that performs a web search based on the user's query.
                It serves as the main entry point for the MCP client request.
                Args:
                    query (str): The search query string provided by the user.
                Returns:
                    str: A human-readable string summarizing the top search results.
                Raises:
                    ValueError: If the query is empty or the API key is missing.
                    :param refined_query:
                """
        if not SERPAPI_KEY:
            raise ValueError("SERPAPI_KEY is not set in the environment variables.")
        data = await make_serpapi_request(refined_query)
        formatted_results = format_search_results(data, refined_query)
        return formatted_results
