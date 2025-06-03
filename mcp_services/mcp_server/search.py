import asyncio
from typing import List, Dict
import logging

from fastapi import FastAPI

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ToolAnnotations
)

from web_search_helper import WebSearchHelper
from pubmed_helper import PubMedHelper

fastApi = FastAPI()

app = Server("mcp-pubmed")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pubmed_helper = PubMedHelper()
web_search_helper = WebSearchHelper()


@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools that this server provides.

    Returns:
        List[Tool]: List of available tools with their descriptions and schemas
    """

    return [
        Tool(
            name="search_pubmed",
            description="Search PubMed literature database and return structured results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (e.g., 'What causes migraines?')"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of articles to retrieve (default: 5, max recommended: 10)"
                    }
                },
                "required": ["query"]
            },
            annotations=ToolAnnotations(
                title="Search PubMed literature database",
                readOnlyHint=True,
                openWorldHint=True
            )
        ),
        Tool(
            name="web_search",
            description="Perform a web search using the SERPAPI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (e.g., 'What is the treatment for diabetes?')"
                    }
                },
                "required": ["query"]
            },
            annotations=ToolAnnotations(
                title="Web Search",
                readOnlyHint=True,
                openWorldHint=True
            )
        )
    ]


@app.call_tool()
async def handle_call_tools(name: str, arguments: Dict) -> List[TextContent]:
    """
    Handle tool execution requests.

    Args:
        name (str): The name of the tool to execute
        arguments (dict): The arguments to pass to the tool

    Returns:
        List[TextContent]: The results of the tool execution
    """
    if name == "search_pubmed":
        return await search_pubmed_literature(arguments)
    elif name == "web_search":
        return await web_search(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def search_pubmed_literature(arguments: Dict) -> List[TextContent]:
    """
    Search PubMed literature database and return structured results.

    This MCP tool method performs a comprehensive literature search by:

    1. Searching PubMed for articles matching the query
    2. Retrieving detailed abstracts for found articles
    3. Combining results into a structured format suitable for LLM processing

    Args:
        arguments (dict): Tool arguments containing:
            - query (str): The search query string
            - max_results (int): Maximum number of articles to retrieve (default: 5)

    Returns:
        List[TextContent]: A list containing the search results formatted as text content.
    """
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    if not query:
        logger.error("Search query cannot be empty")
        return [TextContent(
            text="Error: Search query cannot be empty.",
            type="text"
        )]
    logger.info(f"Starting PubMed search for query: '{query}' (max_results: {max_results})")

    try:
        article_data = await pubmed_helper.search_and_fetch(query, max_results)

        if not article_data.sources_urls:
            logger.warning(f"No PubMed articles found for query: {query}")
            return [TextContent(
                type="text",
                text=f"No PubMed articles found for the search query: '{query}'. Try using different or more general search terms."
            )]

        logger.info(f"Successfully completed PubMed search. Found {len(article_data.sources_urls)} articles.")

        return [TextContent(
            type="text",
            text=article_data.search_results
        )]


    except Exception as e:
        logger.error(f"Unexpected error in PubMed search: {e}")
        return [TextContent(
            type="text",
            text=f"An error occurred while searching PubMed: {str(e)}"
        )]


async def web_search(arguments: Dict) -> List[TextContent]:
    """
    Perform a web search using the SERPAPI.
    Args:
        arguments (dict): Tool arguments containing:
            - query (str): The search query string

    Returns:
        List[TextContent]: A list containing the search results formatted as text content.
    """
    query = arguments.get("query", "")
    if not query:
        logger.error("Search query cannot be empty")
        return [TextContent(
            text="Error: Search query cannot be empty.",
            type="text"
        )]

    logger.info(f"Starting web search for query: '{query}'")
    data = await web_search_helper.search_and_format_results(query)
    if not data.search_results:
        logger.warning(f"No web search results found for query: {query}")
        return [TextContent(
            type="text",
            text=f"No web search results found for the query: '{query}'. Try using different or more general search terms."
        )]
    return [TextContent(
        type="text",
        text=data.search_results
    )]

async def main():
    """
    Main entry point to start the MCP server.

    This function initializes the MCP server and starts listening for requests.
    It also sets up the necessary logging configuration.
    """
    logger.info("Starting MCP PubMed Server...")
    async with stdio_server() as streams:
        # Initialize the MCP server
        await app.run(
            streams[0],  # Input stream
            streams[1],  # Output stream
            app.create_initialization_options()
        )
    logger.info("MCP PubMed Server is running.")


if __name__ == "__main__":
    asyncio.run(main())
