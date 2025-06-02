from typing import List
import logging

import requests
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedArticleData(BaseModel):
    """
    Represents aggregated data from multiple PubMed articles.

    This model contains the combined abstracts/summaries from multiple articles
    along with their source URLs for reference.
    """
    combined_abstracts: str
    source_urls: List[str]


class PubMedHelper:
    """
    Helper class for interacting with the PubMed/NCBI E-utilities API.

    This class provides methods to search for articles and fetch their abstracts
    from the PubMed database using the NCBI E-utilities API.
    """

    def __init__(self):
        self.e_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.e_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.logger = logging.getLogger(__name__)

    async def get_article_ids(self, query: str, max_results: int = 5) -> List[str]:
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

            response = requests.get(self.e_search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            ids = data["esearchresult"]["idlist"]
            self.logger.info(f"Found {len(ids)} PubMed articles for query: {query}")
            return ids

        except requests.RequestException as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
        except KeyError as e:
            self.logger.error(f"Unexpected response format from PubMed: {e}")
            return []

    async def fetch_article_abstracts(self, ids: List[str]) -> PubMedArticleData:
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

            response = requests.get(self.e_fetch_url, params=params, timeout=15)
            response.raise_for_status()

            pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]

            self.logger.info(f"Successfully fetched abstracts for {len(ids)} articles")
            return PubMedArticleData(
                combined_abstracts=response.text,
                source_urls=pubmed_urls
            )

        except requests.RequestException as e:
            self.logger.error(f"Error fetching PubMed abstracts: {e}")
            return PubMedArticleData(
                combined_abstracts=f"Error fetching abstracts: {str(e)}",
                source_urls=[f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]
            )

    async def search_and_fetch(self, query: str, max_results: int = 5) -> PubMedArticleData:
        """
        Convenience method that combines search and fetch operations.

        Args:
            query (str): Search query string
            max_results (int): Maximum number of articles to retrieve

        Returns:
            PubMedArticleData: Combined search and fetch results
        """
        article_ids = await self.get_article_ids(query, max_results)
        if not article_ids:
            return PubMedArticleData(
                combined_abstracts="Error: No articles found for the given query.",
                source_urls=[]
            )

        article_data = await self.fetch_article_abstracts(article_ids)
        response_text = f"PubMed Search Results for: '{query}'\n"
        response_text += f"Found {len(article_data.source_urls)} articles\n\n"
        response_text += "Combined Abstracts:\n"
        response_text += article_data.combined_abstracts
        response_text += f"\n\nSource URLs:\n"
        for i, url in enumerate(article_data.source_urls, 1):
            response_text += f"{i}. {url}\n"

        return PubMedArticleData(
            combined_abstracts=response_text,
            source_urls=article_data.source_urls
        )

