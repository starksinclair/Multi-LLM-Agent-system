from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel

from utils.html_template_generator import HTMLResponseGenerator
from .llm_controller import MedicalLLMController, LLMRole, LLMTask
from .gemini import GeminiLLM
from .deep_seek import DeepSeekLLM
import logging

from .openai import OpenAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generator = HTMLResponseGenerator()
class AgentResponse(BaseModel):
    """
    Represents a standardized response from an individual LLM agent.
    """
    content: str
    provider: str
    model: str


class AgentResponses(BaseModel):
    """
    Gathers responses from different LLM agents involved in processing a question.
    """
    query_refinement: AgentResponse
    research: AgentResponse
    validation: AgentResponse


class AgentResult(BaseModel):
    """
    Encapsulates the complete result of processing a medical question.
    """
    question: str
    web_search_results: Optional[str] = None
    pubmed_results: Optional[str] = None
    agent_responses: AgentResponses
    final_answer: str
    timestamp: datetime


class MultiLLMController:
    """
    Orchestrates a multi-stage process for answering medical questions using different LLM agents.

    This controller manages the lifecycle of a medical question, from initial query refinement
    and web/PubMed search to research, validation, and final answer generation. It assigns specific
    LLM models to different roles (e.g., Query Refiner, Researcher, Validator).
    """

    def __init__(self):
        self.agents: Dict[LLMRole, MedicalLLMController] = {}
        self.setup_agents()

    def setup_agents(self):
        """
        Configures and initializes the LLM agents for each role.

        It attempts to set up agents with specific LLM providers (Gemini, DeepSeek, OpenAI).
        In case of an error during initialization of any specific LLM, it falls back to
        using GeminiLLM for all roles to ensure the system remains operational.
        """
        try:
            gemini_llm = GeminiLLM()
            deep_seek_llm = DeepSeekLLM()
            open_ai_llm = OpenAILLM()
            self.agents = {
                LLMRole.QUERY_REFINER: MedicalLLMController(LLMRole.QUERY_REFINER, gemini_llm),
                LLMRole.RESEARCHER: MedicalLLMController(LLMRole.RESEARCHER, gemini_llm),
                LLMRole.VALIDATOR: MedicalLLMController(LLMRole.VALIDATOR, gemini_llm)
            }
            logger.info("Multi-LLM agent system initialized successfully")
        except Exception as error:
            logger.error(f"Error initializing Multi-LLM agents: {error}")
            fallback_llm = GeminiLLM()
            self.agents = {
                role: MedicalLLMController(role, fallback_llm)
                for role in LLMRole
            }


    async def refine_initial_query(self, query: str) -> AgentResponse:
        """
        Refines the initial user query using the QUERY_REFINER agent to optimize it for web search.

        This method sends the original user question to the QUERY_REFINER LLM agent,
        which is tasked with transforming it into a precise and effective search query.
        It includes basic post-processing to remove potential surrounding quotes from the LLM's output.

        Args:
            query (str): The original user's medical question.

        Returns:
            str: A refined search query string. If the refinement fails or returns
                 an empty response, the original query is returned as a fallback.
        """
        refinement_task = LLMTask(
            task_id="query_refine_001",
            description="Refine initial medical question for search engine",
            prompt=f"Original medical query: '{query}'\n\nRefined medical query for search engine:",
            requires_search_query_refinement=True
        )

        query_refiner_agent = self.agents[LLMRole.QUERY_REFINER]
        try:
            response = await query_refiner_agent.execute_task(refinement_task)
            refined = response.content.strip()
            final_query = refined if refined else query
            return AgentResponse(
                content=final_query,
                provider=response.provider.value,
                model=response.model
            )
        except Exception as error:
            logger.error(f"Gemini API overloaded: {error}. Returning original query.")
            # If the refinement fails or model is overloaded, return the original query
            return  AgentResponse(
            content=query,
            provider=query_refiner_agent.llm.get_provider().value,
            model=query_refiner_agent.llm.model
        )
            # logger.error(f"Error during query refinement: {error}")
            # return query

    async def process_medical_question(self, question: str, web_search_results: Optional[str] = None,
                                       pubmed_results: Optional[str] = None) -> AgentResult:
        """
        Processes a medical question through a multi-stage pipeline involving LLM agents
        for research, and validation.

        This asynchronous method orchestrates the following steps:

        1. Passes the original question along with both web search results and PubMed results
           to a RESEARCHER agent to synthesize relevant medical information from multiple sources
           aiming for a concise output (approx. 1000 tokens).

        2. Sends the RESEARCHER's output to a VALIDATOR agent to ensure safety,
           accuracy, disclaimers, and formats the final answer into well-structured HTML with Tailwind (approx. 1000 tokens).

        3. Compiles all intermediate and final responses into an `AgentResult` object.

        Args:
            question (str): The medical question to be processed.
            web_search_results (Optional[str]): Pre-fetched web search results.
            pubmed_results (Optional[str]): Pre-fetched PubMed search results.
                If None, no PubMed data will be included in the research.

        Returns:
            AgentResult: A comprehensive object containing the original question,
                         search results, PubMed results, responses from each agent,
                         and the final validated HTML answer.

        Raises:
            RuntimeError: If the research response is empty or invalid.
            Exception: If any error occurs during the processing stages.
        """
        logger.info(f"Processing medical question: {question}")

        research_prompt = f"""Analyze this medical question using the comprehensive search information provided from multiple sources:

        You have a strict limit of approximately **1000 tokens** for the final output. Adjust detail level, brevity, and formatting accordingly to fit this constraint.

        Original Question: {question}
        """

        if web_search_results:
            research_prompt += f"""

        WEB SEARCH RESULTS:
        {web_search_results}
        """

        if pubmed_results:
            research_prompt += f"""

        PUBMED LITERATURE RESULTS:
        {pubmed_results}
        """

        research_prompt += """

        RESEARCH INSTRUCTIONS:
        Based on the search results from both web sources and medical literature (if available), extract and synthesize the most relevant and reliable medical information. 

        Prioritize information in this order:
        1. Peer-reviewed medical literature (PubMed results)
        2. Reputable medical sources (Mayo Clinic, WebMD, NIH, medical journals)
        3. Other credible health websites

        Focus on identifying key points about:
        - Symptoms or conditions mentioned
        - Potential causes and risk factors
        - Treatment options and management strategies
        - When to seek medical care
        - Prevention measures (if applicable)

        IMPORTANT GUIDELINES:
        - Cross-reference information between web and literature sources when possible
        - If sources contradict each other, mention this and favor peer-reviewed literature
        - If search results are inconclusive, contradictory, or if information on a particular key point is scarce, state this clearly
        - Do not invent information or make assumptions beyond what the sources provide
        - If the question is outside the scope of general medical knowledge, state that appropriately
        - Emphasize that this information is for educational purposes only

        Synthesize the information from both sources into a coherent, evidence-based response."""

        try:
            research_task = LLMTask(
                task_id="research_001",
                description="Research medical question using web and literature sources",
                prompt=research_prompt,
                system_prompt=self.agents[LLMRole.RESEARCHER].system_prompts[LLMRole.RESEARCHER],
                requires_search=True
            )

            research_response = await self.agents[LLMRole.RESEARCHER].execute_task(research_task)
            if not research_response or not research_response.content.strip():
                raise RuntimeError("Empty research response — cannot proceed to validation")

            html_wrapper = generator.generate_html()
            validation_task = LLMTask(
                task_id="validation_001",
                description="Validate final medical response",
                prompt=f"""
                        You are validating a medical response to ensure it meets safety and quality standards before presenting it to users.

                        You have a strict limit of approximately **1000 tokens** for the final output. Adjust the level of detail, brevity, and formatting accordingly to fit within this limit.

                        Here is the draft response based on web search and literature review:

                        {research_response.content}

                        REQUIREMENTS:
                        1. Format your final output using HTML only — no markdown or explanations.
                        2. Use Tailwind CSS classes and structure your content freely as appropriate for the question.
                        3. Wrap your output with this exact template (do NOT change structure or classes):

                        {html_wrapper}

                        4. Replace {{YOUR_BODY_HTML_HERE}} with your validated content using valid HTML elements.
                        5. Ensure the disclaimers remain exactly as shown at the top and bottom.

                        Return only the final rendered HTML.
                        """,
                system_prompt=self.agents[LLMRole.VALIDATOR].system_prompts[LLMRole.VALIDATOR],
            )
            validation_response = await self.agents[LLMRole.VALIDATOR].execute_task(validation_task)
            logger.info("Medical question processing completed")

            return AgentResult(
                question=question,
                web_search_results=web_search_results,
                pubmed_results=pubmed_results,
                agent_responses=AgentResponses(
                    query_refinement=AgentResponse(
                        content=question,
                        provider=self.agents[LLMRole.QUERY_REFINER].llm.get_provider().value,
                        model=self.agents[LLMRole.QUERY_REFINER].llm.model
                    ),
                    research=AgentResponse(
                        content=research_response.content,
                        provider=research_response.provider.value,
                        model=research_response.model
                    ),
                    validation=AgentResponse(
                        content=validation_response.content,
                        provider=validation_response.provider.value,
                        model=validation_response.model
                    ),
                ),
                final_answer=validation_response.content,
                timestamp=datetime.now(),
            )
        except Exception as error:
            logger.error(f"Model failed or is overloaded: {error}")
            message = "The model is currently overloaded due to a high volume of requests. Please try again during off peak hours."
            return AgentResult(
                question=question,
                web_search_results=web_search_results,
                pubmed_results=pubmed_results,
                agent_responses=AgentResponses(
                    query_refinement=AgentResponse(
                        content=question,
                        provider=self.agents[LLMRole.QUERY_REFINER].llm.get_provider().value,
                        model=self.agents[LLMRole.QUERY_REFINER].llm.model
                    ),
                    research=AgentResponse(
                        content=message,
                        provider="N/A",
                        model="N/A"
                    ),
                    validation=AgentResponse(
                        content=message,
                        provider="N/A",
                        model="N/A"
                    ),
                ),
                final_answer=message,
                timestamp=datetime.now(),
            )
