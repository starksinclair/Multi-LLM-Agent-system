"""
MCP Web Search Client Module

This module provides a client interface for processing queries through a multi-LLM
medical agent system with MCP web search capabilities.
"""
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client

from llm_agents.multi_llm_controller import MultiLLMController, AgentResult

controller = MultiLLMController()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for interacting with the Multi-LLM Controller to process medical questions.

    This class serves as an interface for external components to submit medical
    queries and receive comprehensive results processed by a pipeline of LLM agents.
    Example:
        >>> async def example():
        ...     client = MCPClient()
        ...     result = await client.run("What are the symptoms of diabetes?")
        ...     return result
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        logger.info(f"Initializing MCP Web Search Client")

    async def connect_to_server(self, server_script_path: str):
        """
           Connects to the MCP server using the provided script path.

           This method initializes a ClientSession and connects to the specified
           MCP server script, allowing for interaction with the MultiLLMController.

           Args:
               server_script_path (str): The file path to the MCP server script.

           Returns:
               list: A list of tool objects available from the connected MCP server.

           Raises:
               Exception: If the connection to the MCP server fails.
           """
        try:
            params = StdioServerParameters(
                command='python',
                args=[server_script_path],
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio_transport[0], stdio_transport[1]))

            await self.session.initialize()
            response = await self.session.list_tools()
            tools = response.tools
            print(f"Connected to MCP server with tools: {[tool.name for tool in tools]}")

            return tools

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise


    async def run(self, query: str) -> AgentResult:
        """
        Processes a medical question using the MultiLLMController.

        This asynchronous method takes a user's medical query, passes it to the
        MultiLLMController for multi-stage processing (including refinement,
        research, and validation), and returns the final structured result.

        Args:
            query (str): The medical question string provided by the user.

        Returns:
            AgentResult: A Pydantic object containing the comprehensive results
                         of the medical question processing, including agent responses
                         and the final formatted answer.

        Raises:
            ValueError: If the input query string is empty.
            Exception: Propagates any exceptions that occur during the MultiLLMController's
                       processing of the medical question.
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        try:
            await self.connect_to_server("mcp_services/mcp_server/search.py")

            refined_query = await controller.refine_initial_query(query)
            logger.info(f"Refined query: {refined_query}")

            pubmed_task = self.session.call_tool(
                name="search_pubmed",
                arguments={"query": refined_query.content, "max_results": 5}
            )
            web_search_task = self.session.call_tool(
                name="web_search",
                arguments={"query": refined_query.content}
            )
            pubmed_search_results, web_search_results = await asyncio.gather(
                pubmed_task,
                web_search_task
            )

            logger.info(f"PubMed search completed successfully")
            return await controller.process_medical_question(refined_query.content, web_search_results.content[0].text, pubmed_search_results.content[0].text)

        except Exception as e:
            logger.error(f"Error in run method: {e}")
            raise
        finally:
            await self.close()

    async def close(self):
        """Close the client session and cleanup resources."""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.error(f"Error closing client: {e}")


