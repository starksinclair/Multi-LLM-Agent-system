from typing import Dict
from datetime import datetime
from pydantic import BaseModel

from mcp_services.mcp_server.mcp_web_search_server import MCPWebSearchServer, SearchResult
from .llm_controller import MedicalLLMController, LLMRole, LLMTask
from .gemini import GeminiLLM
from .deep_seek import DeepSeekLLM
import logging

from .openai import OpenAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    search_results: SearchResult
    agent_responses: AgentResponses
    final_answer: str
    timestamp: datetime


class MultiLLMController:
    """
    Orchestrates a multi-stage process for answering medical questions using different LLM agents.

    This controller manages the lifecycle of a medical question, from initial query refinement
    and web search to research, validation, and final answer generation. It assigns specific
    LLM models to different roles (e.g., Query Refiner, Researcher, Validator) to leverage
    their unique strengths.
    """
    def __init__(self):
        self.mcp_server = MCPWebSearchServer()
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
                LLMRole.RESEARCHER: MedicalLLMController(LLMRole.RESEARCHER, deep_seek_llm),
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

    async def refine_initial_query(self, query: str) -> str:
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
        logger.info(f"Refining initial query: {query}")
        refinement_task = LLMTask(
            task_id="query_refine_001",
            description="Refine initial medical question for search engine",
            prompt=f"Original medical query: '{query}'\n\nRefined medical query for search engine:",
            requires_search_query_refinement=True
        )

        query_refiner_agent = self.agents[LLMRole.QUERY_REFINER]
        logger.info(f"Refiner agent: {query_refiner_agent}")
        refinement_response = await query_refiner_agent.execute_task(refinement_task)
        if not refinement_response or not refinement_response.content:
            logger.warning("Refinement response is empty, returning original query")
            return query

        refined_query = refinement_response.content.strip()
        # remove quotes if Gemini wraps the output in them
        if refined_query.startswith('"') and refined_query.endswith('"'):
            refined_query = refined_query[1:-1]

        logger.info(f"Initial query refined: '{query}' -> '{refined_query}'")
        return refined_query if refined_query else query

    async def process_medical_question(self, question: str) -> AgentResult:
        """
        Processes a medical question through a multi-stage pipeline involving LLM agents
        for query refinement, research, and validation.

        This asynchronous method orchestrates the following steps:

        1. Refines the initial user question into an optimized search query using the QUERY_REFINER agent.

        2. Executes a web search using the refined query via `self.mcp_server` which is the custom MCP Server.

        3. Passes the original question and search results to a RESEARCHER agent
           to summarize relevant medical information.

        4. Sends the RESEARCHER's output to a VALIDATOR agent to ensure safety,
           accuracy, disclaimers, and proper HTML formatting.
        5. Compiles all intermediate and final responses into an `AgentResult` object.

        Args:
            question (str): The medical question to be processed.

        Returns:
            AgentResult: A comprehensive object containing the original question,
                         search results, responses from each agent, and the final
                         validated HTML answer.
        """
        logger.info(f"Processing medical question: {question}")
        refined_initial_query = await self.refine_initial_query(question)
        search_results = await self.mcp_server.run(f"medical {refined_initial_query}")
        research_task = LLMTask(
            task_id="research_001",
            description="Research medical question",
            prompt=f"""Analyze this medical question and the *refined* search information provided:
                   You have a strict limit of approximately **1000 tokens** for the final output. Adjust detail level, brevity, and formatting accordingly to fit this constraint.
                     
                   Original Question: {question}

                    Refined Medical Search Information:
                    {refined_initial_query}

                    Based on the search results, extract and summarize the most relevant and reliable medical information. 
                    Focus on information from reputable medical sources like Mayo Clinic, WebMD, NIH, or medical journals.
                    
                    Identify key points about:
                    - Symptoms or conditions mentioned
                    - Potential causes
                    - Treatment options
                    - When to seek medical care
                    - If the search results are inconclusive, contradictory, or if information on a particular key point is scarce, state this clearly. Do not invent information. If the question is outside the scope of general medical knowledge, state that appropriately.
                    
                    Remember to emphasize that this information is for educational purposes only.""",
            system_prompt=self.agents[LLMRole.RESEARCHER].system_prompts[LLMRole.RESEARCHER],
            requires_search=True
        )
        research_response = await self.agents[LLMRole.RESEARCHER].execute_task(research_task, search_results)
        logger.info(f"Research response: {research_response.content}")

        validation_task = LLMTask(
            task_id="validation_001",
            description="Validate final medical response",
            prompt=f"""You are validating a medical response to ensure it meets safety and quality standards before presenting it to users.

                    You have a strict limit of approximately **1000 tokens** for the final output. Adjust detail level, brevity, and formatting accordingly to fit this constraint.
                    
                    Here is the draft response:
                    
                    {research_response.content}
                    
                    Perform the following checks:
                    1. Do **not** provide specific medical diagnoses or treatment advice.
                    2. Include a strong disclaimer advising users to consult a qualified healthcare professional.
                    3. Ensure the content is safe, medically accurate, and educational — not misleading or harmful.
                    4. Use clear, concise language suitable for a general audience. Briefly explain any medical terms if needed.
                    5. Keep the content focused and avoid unnecessary elaboration.
                    
                    Once validated, transform the response into an **HTML snippet with in-line CSS styles** for display in a user interface, just include the html don't add ```html.
                    
                    ### Formatting instructions:
                    - Use `<h2>` tags for section headings like **Symptoms**, **Potential Causes**, **Treatment Options**, **When to Seek Medical Care**
                    - Use bullet points (`<ul><li>`) for readability where appropriate
                    - Place the following **strong disclaimer** at both the top and bottom:
                    
                    > <strong>This information is for educational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.</strong>
                    
                    Only return the final HTML output. Do **not** include explanation or additional commentary.""",

            system_prompt=self.agents[LLMRole.VALIDATOR].system_prompts[LLMRole.VALIDATOR],
        )
        validation_response = await self.agents[LLMRole.VALIDATOR].execute_task(validation_task)
        logger.info(f"Validation response: {validation_response.content}")
        logger.info("Medical question processing completed")
        return AgentResult(
            question=question,
            search_results=search_results,
            agent_responses=AgentResponses(
                query_refinement=AgentResponse(
                    content=refined_initial_query,
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
