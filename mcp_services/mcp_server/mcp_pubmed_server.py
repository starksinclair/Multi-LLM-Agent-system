from typing import List
import logging

from fastapi import FastAPI
import requests
from mcp.server import FastMCP
from pydantic import BaseModel

app = FastAPI()
mcp = FastMCP("mcp_pubmed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

E_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
E_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedArticleData(BaseModel):
    """
    Represents aggregated data from multiple PubMed articles.

    This model contains the combined abstracts/summaries from multiple articles
    along with their source URLs for reference.
    """
    combined_abstracts: str
    source_urls: List[str]


class PubMedSearchResult(BaseModel):
    """
    Represents a structured result from a PubMed search operation.

    This is the final output model that contains all search metadata
    and the processed article content for consumption by LLM agents.
    """
    tool: str
    query: str
    answer: str
    sources: List[str]
    total_results: int


async def get_pubmed_article_ids(query: str, max_results: int = 5) -> List[str]:
    """
    Search PubMed and retrieve a list of article IDs matching the query.

    Uses the NCBI E-utilities E-search API to find relevant PubMed articles
    and returns their unique identifiers (PMIDs).

    Args:
        query (str): Search query string (e.g., "diabetes treatment")
        max_results (int): Maximum number of article IDs to return (default: 5)

    Returns:
        List[str]: List of PubMed IDs (PMIDs) as strings

    Raises:
        requests.RequestException: If the API request fails
        KeyError: If the response format is unexpected
    """
    try:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        }
        response = requests.get(E_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        ids = data["esearchresult"]["idlist"]
        logger.info(f"Found {len(ids)} PubMed articles for query: {query}")
        return ids

    except requests.RequestException as e:
        logger.error(f"Error searching PubMed: {e}")
        return []
    except KeyError as e:
        logger.error(f"Unexpected response format from PubMed: {e}")
        return []


async def fetch_pubmed_article_abstracts(ids: List[str]) -> PubMedArticleData:
    """
    Fetch full article abstracts for the given PubMed IDs.

    Uses the NCBI E-utilities E-fetch API to retrieve detailed article information
    including abstracts, which are then combined into a single text block.

    Args:
        ids (List[str]): List of PubMed IDs to fetch abstracts for

    Returns:
        PubMedArticleData: Contains combined abstracts and source URLs

    Raises:
        requests.RequestException: If the API request fails
    """
    if not ids:
        return PubMedArticleData(combined_abstracts="", source_urls=[])

    try:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "text",
            "rettype": "abstract"
        }
        response = requests.get(E_FETCH_URL, params=params, timeout=15)  # Added timeout
        response.raise_for_status()

        pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]

        logger.info(f"Successfully fetched abstracts for {len(ids)} articles")
        return PubMedArticleData(
            combined_abstracts=response.text,
            source_urls=pubmed_urls
        )

    except requests.RequestException as e:
        logger.error(f"Error fetching PubMed abstracts: {e}")
        return PubMedArticleData(
            combined_abstracts=f"Error fetching abstracts: {str(e)}",
            source_urls=[f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]
        )


class PubMedSearchMCPServer:
    """
    MCP Server for handling PubMed literature search operations.

    This server provides MCP tool functionality for searching PubMed database
    and retrieving scientific literature abstracts. It's designed to be used
    by LLM agents that need access to current medical/scientific research.

    The server handles the complete workflow:
    1. Search PubMed for relevant articles
    2. Fetch detailed abstracts
    3. Return structured results for LLM consumption

    Attributes:
        tool_name (str): Identifier for this MCP

    Example:
            >>> async def example():
            ...     server = PubMedSearchMCPServer()
            ...     results = await server.search_pubmed_literature("diabetes treatment")
            ...     return results
    """

    def __init__(self, tool_name: str = "pubmed_search"):
        self.tool_name = tool_name
        logger.info(f"Initialized PubMed MCP Server with tool name: {tool_name}")

    @mcp.tool()
    async def search_pubmed_literature(self, query: str, max_results: int = 5) -> PubMedSearchResult:
        """
        Search PubMed literature database and return structured results.

        This MCP tool method performs a comprehensive literature search by:

        1. Searching PubMed for articles matching the query
        2. Retrieving detailed abstracts for found articles
        3. Combining results into a structured format suitable for LLM processing

        Args:
            query (str): The search query string (e.g., "CRISPR gene editing", "COVID-19 treatment")
            max_results (int): Maximum number of articles to retrieve (default: 5, max recommended: 20)

        Returns:
            PubMedSearchResult: Structured search results containing:

                - tool: The tool identifier
                - query: Original search query
                - answer: Combined abstracts from found articles
                - sources: List of PubMed URLs for reference
                - total_results: Number of articles found
        """

        if not query:
            logger.error("Search query cannot be empty")
            return PubMedSearchResult(
                tool=self.tool_name,
                query=query,
                answer="Search query cannot be empty.",
                sources=[],
                total_results=0
            )
        logger.info(f"Starting PubMed search for query: '{query}' (max_results: {max_results})")

        try:
            article_ids = await get_pubmed_article_ids(query, max_results)

            if not article_ids:
                logger.warning(f"No PubMed articles found for query: {query}")
                return PubMedSearchResult(
                    tool=self.tool_name,
                    query=query,
                    answer=f"No PubMed articles found for the search query: '{query}'. Try using different or more general search terms.",
                    sources=[],
                    total_results=0
                )

            article_data = await fetch_pubmed_article_abstracts(article_ids)

            result = PubMedSearchResult(
                tool=self.tool_name,
                query=query,
                answer=article_data.combined_abstracts,
                sources=article_data.source_urls,
                total_results=len(article_data.source_urls)
            )

            logger.info(f"Successfully completed PubMed search. Found {result.total_results} articles.")
            return result

        except Exception as e:
            logger.error(f"Unexpected error in PubMed search: {e}")
            return PubMedSearchResult(
                tool=self.tool_name,
                query=query,
                answer=f"An error occurred while searching PubMed: {str(e)}",
                sources=[],
                total_results=0
            )
