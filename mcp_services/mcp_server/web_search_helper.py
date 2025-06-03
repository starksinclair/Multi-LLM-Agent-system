import os
import logging
import json
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests
from pydantic import BaseModel

load_dotenv()
mcp = FastMCP("web-search")
URL = "https://serpapi.com/search"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """
    Represents aggregated data from the search result.

    This model contains the search results including titles, URLs, snippets
    along with their source URLs for reference.

    """
    search_results: str
    sources_urls: list[str]


class WebSearchHelper:
    """
    Helper class for performing web searches using SerpAPI.

    This class provides methods to search the web and return structured results
    that can be consumed by LLM agents.
    """

    def __init__(self, api_key: Optional[str] = None):
        # self.serpapi_key =  os.environ.get("SERP_API_KEY")
        self.serpapi_key = api_key or os.environ.get("SERP_API_KEY")
        if not self.serpapi_key:
            raise ValueError("SerpAPI key must be provided.")
        logger.info("Initialized WebSearchHelper with SerpAPI key.")

    async def search_and_format_results(self, query: str) -> SearchResult:
        """
        Performs an HTTP GET request to the SerpAPI Google search API.

        This asynchronous function constructs the request parameters, including the
        search query, API key, desired engine, number of results, and safe search settings.
        It then executes the request and parses the JSON response.

        Args:
            query (str): The search query string.

        Returns:
            SearchResult: Parsed JSON response from SerpAPI containing search results
                          and metadata, structured for easy consumption by LLM agents.

        Raises:
            Exception: If the HTTP response status code is not 200 or if an error
                       message is returned by the SerpAPI.
        """
        params = {
            'q': query,
            "api_key": self.serpapi_key,
            "engine": "google",
            "num": 5,
            "safe": "active",
        }

        response = requests.get(URL, params=params)
        data = response.json()
        if response.status_code != 200:
            raise Exception(f"Error fetching data from SerpAPI: {data.get('error', 'Unknown error')}")
        results = []


        for i, result in enumerate(data.get("organic_results", []), 1):
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "")
            source = result.get("source", "")

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": source
            })

        return SearchResult(
            search_results=json.dumps(results),
            sources_urls=[result.get("link", "") for result in data.get("organic_results", [])],
        )