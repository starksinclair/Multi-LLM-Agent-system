import logging
import requests
import xml.etree.ElementTree as ET
import json

from typing import List
from pydantic import BaseModel

from web_search_helper import SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedResult(BaseModel):
    """
    Represents a single parsed PubMed article with key details,
    consistent with web search results for frontend display.
    """
    title: str
    url: str
    snippet: str  # Stores the abstract content
    source: str  # Stores the journal title or "PubMed"


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

    async def fetch_article_abstracts(self, xml_string: str) -> List[PubMedResult]:
        """
        Parses an XML string response from PubMed E-fetch API into structured PubMedResult objects.

        This method extracts the Article Title, PMID (for URL construction), Journal Title,
        and combines all AbstractText sections into a single snippet for each article found in the XML.

        Args:
            xml_string (str): The XML response content from the NCBI E-fetch API.

        Returns:
            List[PubMedResult]: A list of structured PubMed article data.
        """
        parsed_articles: List[PubMedResult] = []

        try:
            # Parse the XML string
            root = ET.fromstring(xml_string)

            for pubmed_article in root.findall('PubmedArticle'):
                pmid_element = pubmed_article.find('.//MedlineCitation/PMID')
                pmid = pmid_element.text if pmid_element is not None else 'N/A'

                article_title_element = pubmed_article.find('.//Article/ArticleTitle')
                title = article_title_element.text if article_title_element is not None else 'N/A'

                abstract_texts = []
                for abs_text_elem in pubmed_article.findall('.//Abstract/AbstractText'):
                    if abs_text_elem.text:
                        label = abs_text_elem.get('Label')
                        text_content = abs_text_elem.text.strip()
                        if label and text_content:
                            abstract_texts.append(f"**{label.capitalize()}**: {text_content}")  # Example: Bold label
                        elif text_content:
                            abstract_texts.append(text_content)
                snippet = "\n\n".join(abstract_texts).strip() if abstract_texts else 'No abstract available.'

                journal_title_element = pubmed_article.find('.//Journal/Title')
                journal_title = journal_title_element.text if journal_title_element is not None else 'PubMed Journal'

                article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != 'N/A' else 'N/A'

                if title != 'N/A' and article_url != 'N/A' and pmid != 'N/A':
                    parsed_articles.append(PubMedResult(
                        title=title,
                        url=article_url,
                        snippet=snippet,
                        source=journal_title
                    ))
        except ET.ParseError as e:
            self.logger.error(f"Error parsing PubMed XML: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during PubMed XML parsing: {e}")
            return []

        return parsed_articles

    async def search_and_fetch(self, query: str, max_results: int = 5) -> SearchResult:
        """
        Convenience method that combines search and fetch operations for PubMed.

        This method first searches PubMed for article IDs, then fetches their full XML data,
        parses it into a list of structured PubMedResult objects, and finally converts this
        list into a JSON string suitable for direct consumption by the frontend and LLM Agents.

        Args:
            query (str): Search query string
            max_results (int): Maximum number of articles to retrieve

        Returns:
            str: A JSON string representing a list of PubMedResult objects.
                 Returns an empty JSON array string if no articles are found or an error occurs.

        Raises:
            requests.RequestException: If the API request fails during search or fetch
            Exception: If there is an unexpected error during XML parsing or article processing
        """
        article_ids = await self.get_article_ids(query, max_results)
        if not article_ids:
            return SearchResult(
                search_results="No articles found for the given query.",
                sources_urls=[]
            )

        try:
            params = {
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
                "rettype": "abstract"
            }

            response = requests.get(self.e_fetch_url, params=params, timeout=20)  # Increased timeout slightly
            response.raise_for_status()

            xml_string = response.text

            parsed_articles = await self.fetch_article_abstracts(xml_string)

            self.logger.info(f"Successfully fetched and parsed PubMed XML for {len(parsed_articles)} articles")

            data = json.dumps([article.model_dump() for article in parsed_articles])
            return SearchResult(
                search_results=data,
                sources_urls=[article.url for article in parsed_articles]
            )

        except requests.RequestException as e:
            self.logger.error(f"Error fetching PubMed XML: {e}")
            return SearchResult(
                search_results="Error fetching PubMed articles. Please try again later.",
                sources_urls=[]
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in PubMed XML processing or article parsing: {e}")
            return SearchResult(
                search_results="An unexpected error occurred while processing PubMed articles.",
                sources_urls=[]
            )