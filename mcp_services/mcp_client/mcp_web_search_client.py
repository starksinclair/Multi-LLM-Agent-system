"""
MCP Web Search Client Module

This module provides a client interface for processing queries through a multi-LLM
medical agent system with MCP web search capabilities.
"""
import logging
from dotenv import load_dotenv
from llm_agents.multi_llm_controller import MultiLLMController, AgentResult

controller = MultiLLMController()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPWebSearchClient:
    """
    Client for interacting with the Multi-LLM Controller to process medical questions.

    This class serves as an interface for external components to submit medical
    queries and receive comprehensive results processed by a pipeline of LLM agents.
    Example:
        >>> async def example():
        ...     client = MCPWebSearchClient()
        ...     result = await client.run("What are the symptoms of diabetes?")
        ...     return result
    """

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

        data = await controller.process_medical_question(query)
        return data
