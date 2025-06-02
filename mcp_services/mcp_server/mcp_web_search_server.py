import os
from typing import Dict, Any
import logging

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests
from pydantic import BaseModel

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
load_dotenv()
mcp = FastMCP("web-search")
URL = "https://serpapi.com/search"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchResult(BaseModel):
    """
    Represents the structured result from a Web search operation.

    This is the final output model that contains all search metadata
    and the processed search results for consumption by LLM agents.
    """
    tool: str
    query: str
    answer: list[Dict[str, Any]] | str
    sources: list[str]
    total_results: int

class SearchResult(BaseModel):
    """
    Represents aggregated data from the search result.

    This model contains the search results including titles, URLs, snippets
    along with their source URLs for reference.

    """
    search_results: list[Dict[str, Any]]
    sources_urls: list[str]

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


def format_search_results(data: dict) -> SearchResult:
    """
    Converts the raw JSON response from SerpAPI into a structured SearchResult model.

    This function extracts relevant information like titles, URLs, snippets, and sources
    from the SerpAPI response's "organic_results" and encapsulates them into a
    Pydantic SearchResult object.

    Args:
        data (dict): The JSON response dictionary from SerpAPI.

    Returns:
        SearchResult: A Pydantic model containing the search results, including titles,urls, snippets, along with their source urls.
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
        search_results=results,
        sources_urls=[result.get("link", "") for result in data.get("organic_results", [])]
    )


class MCPWebSearchServer:
    """
    Manages web search operations by integrating with SerpAPI via the FastMCP framework.

    This class provides access to the Web Search MCP Server Tool run method which handles search requests and returns
    formatted results through the MCP tool interface.


    Example:
        >>> async def example():
        ...     server = MCPWebSearchServer()
        ...     results = await server.run("What is the treatment for diabetes?")
        ...     return results
    """

    def __init__(self, tool_name: str = "web_search"):
        self.tool_name = tool_name
        logger.info(f"Initialized Web Search MCP Server with tool name: {tool_name}")

    @mcp.tool()
    async def run(self, refined_query: str) -> str | WebSearchResult:
        """
        MCP tool method that performs a web search based on the refined query.

        This method serves as the main entry point for MCP client requests,
        executing web searches through SerpAPI and returning formatted results.

        Args:
            refined_query (str): The refined search query string to be executed.
                                Must be a non-empty string.

        Returns:
            str | SearchResult: A human-readable string summarizing the top search
                               results, or a SearchResult object containing structured
                               search data.

        Raises:
            ValueError: If the refined_query is empty or if SERPAPI_KEY is not set
                       in the environment variables.


        """
        if not SERPAPI_KEY:
            raise ValueError("SERPAPI_KEY is not set in the environment variables.")
        if not refined_query:
            return WebSearchResult(
                tool=self.tool_name,
                query=refined_query,
                answer="Search query cannot be empty.",
                sources=[],
                total_results=0
            )
        data = await make_serpapi_request(refined_query)
        formatted_results = format_search_results(data)
        return WebSearchResult(
        tool=self.tool_name,
        query=refined_query,
        answer=formatted_results.results,
        sources=formatted_results.sources,
        total_results=len(formatted_results.results),
    )
