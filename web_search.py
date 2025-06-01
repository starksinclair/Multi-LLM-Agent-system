import logging
import os
from google import genai
import requests
from dotenv import load_dotenv
from google.genai import types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# mcp = FastMCP("web-search")
# URL = "https://serpapi.com/search"
from mcp.mcp_server import MCPServer

server = MCPServer()
load_dotenv()

class SearchResult(BaseModel):
    tool: str
    query: str
    answer: str
    sources: list[str]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def refine_search_query(query: str) -> str:
    """
    Refines a medical search query to improve search results for SerpAPI,
    using the Gemini API for medical term precision and search optimization.

    Args:
        query (str): The original medical search query.

    Returns:
        str: The refined medical search query optimized for SerpAPI.
    """

    client = genai.Client(api_key="AIzaSyAYLsAGikd_pbdgJGHfsDABHuE76efHqLg")

    # Construct the prompt for Gemini, emphasizing medical context and SerpAPI optimization
    medical_refinement_prompt = (
        f"Refine the following medical search query to make it more precise and effective "
        f"for a search engine like Google (which SerpAPI queries). "
        f"Focus on using accurate medical terminology, adding relevant keywords, "
        f"and formulating it for direct search result relevance (e.g., symptoms, treatments, drug info, disease mechanisms). "
        f"The output should be only the refined query string, with no additional text or explanation.\n\n"
        f"Original medical query: '{query}'\n\n"
        f"Refined medical query for search engine:"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Using the specified model
            config=types.GenerateContentConfig(
                # System instruction to ensure Gemini behaves as a medical expert for this task
                system_instruction="You are an expert medical search query optimizer. Your goal is to transform user questions into precise and effective search queries for medical research."
            ),
            contents=medical_refinement_prompt
        )

        refined_query = response.text.strip()
        print(refined_query)
        logger.info(f"Refined query: {refined_query}")
        if not refined_query:
            raise ValueError("Refined query is empty. Please check the input query.")

        # Basic post-processing: remove quotes if Gemini wraps the output in them
        if refined_query.startswith('"') and refined_query.endswith('"'):
            refined_query = refined_query[1:-1]

        return refined_query
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return query.strip()
def refine_search_result(data) -> str | SearchResult:

    return
# def refine_search



class WebSearchTool:
    async def run(self, query: str) -> str | SearchResult:

        if not query:
            raise ValueError("Query cannot be empty.")

        refined_query = refine_search_query(query)
        data = await server.run(refined_query)
        if isinstance(data, SearchResult):
            return data
        else:
            return SearchResult(
                tool="WebSearchTool",
                query=query,
                answer=data,
                sources=[]
            )