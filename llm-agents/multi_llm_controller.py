import asyncio
from typing import Dict, Any

from mcp_services.mcp_web_search_server import  MCPWebSearchServer
from llm_controller import MedicalLLMController, LLMRole, LLMTask
from gemini import GeminiLLM
from deep_seek import DeepSeekLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLLMController:
    def __init__(self):
        self.mcp_server =  MCPWebSearchServer()
        self.agents: Dict[LLMRole, MedicalLLMController] = {}
        self.setup_agents()

    def setup_agents(self):
        try:
            gemini_llm = GeminiLLM()
            deep_seek_llm = DeepSeekLLM()
            self.agents = {
                LLMRole.RESEARCHER: MedicalLLMController(LLMRole.RESEARCHER, gemini_llm, self.mcp_server),
                LLMRole.ANALYZER: MedicalLLMController(LLMRole.ANALYZER, deep_seek_llm, self.mcp_server),
                LLMRole.SYNTHESIZER: MedicalLLMController(LLMRole.SYNTHESIZER, gemini_llm, self.mcp_server),
                LLMRole.VALIDATOR: MedicalLLMController(LLMRole.VALIDATOR, deep_seek_llm, self.mcp_server)
            }
            logger.info("Multi-LLM agent system initialized successfully")
        except Exception as error:
            logger.error(f"Error initializing Multi-LLM agents: {error}")
            fallback_llm = GeminiLLM()
            self.agents = {
                role: MedicalLLMController(role, fallback_llm, self.mcp_server)
                for role in LLMRole
            }

    async def refine_initial_query(self, query: str) -> str:
        """
        Refines the initial user query using the QUERY_REFINER agent.
        """
        refinement_task = LLMTask(
            task_id="query_refine_001",
            description="Refine initial medical question for search engine",
            prompt=f"Original medical query: '{query}'\n\nRefined medical query for search engine:",
            requires_search_query_refinement=True
        )

        query_refiner_agent = self.agents[LLMRole.QUERY_REFINER]
        refinement_response = await query_refiner_agent.execute_task(refinement_task)

        refined_query = refinement_response.content.strip()
        # Basic post-processing: remove quotes if Gemini wraps the output in them
        if refined_query.startswith('"') and refined_query.endswith('"'):
            refined_query = refined_query[1:-1]

        logger.info(f"Initial query refined: '{query}' -> '{refined_query}'")
        return refined_query if refined_query else query

    async def process_medical_question(self, question: str) -> Dict[str, Any]:
        """
        Process a medical question using multiple LLM agents.

        Args:
            question (str): The medical question to be processed.

        Returns:
            Dict[LLMRole, str]: Responses from each LLM agent categorized by role.
        """
        logger.info(f"Processing medical question: {question}")
        refined_initial_query = await self.refine_initial_query(question)
        search_results = await self.mcp_server.run(f"medical {refined_initial_query}")
        research_task = LLMTask(
            task_id="research_001",
            description="Research medical question",
            prompt=f"""Analyze this medical question and the *refined* search information provided:
            
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
                    
                    Remember to emphasize that this information is for educational purposes only.""",
            system_prompt=self.agents[LLMRole.RESEARCHER].system_prompts[LLMRole.RESEARCHER],
            requires_search=True
        )
        research_response = await self.agents[LLMRole.RESEARCHER].execute_task(research_task, search_results)
        logger.info(f"Research response: {research_response.content}")

        analysis_task = LLMTask(
            task_id="analysis_001",
            description="Analyze medical information",
            prompt=f"""Review the following medical research findings:

                    {research_response.content}
                    
                    Analyze this information for:
                    1. Accuracy and consistency
                    2. Source reliability
                    3. Completeness of information
                    4. Any potential gaps or concerns
                    5. Quality of evidence presented
                    
                    Provide your analysis and suggest any improvements or additional considerations.""",
            system_prompt=self.agents[LLMRole.ANALYZER].system_prompts[LLMRole.ANALYZER],
        )
        analysis_response = await self.agents[LLMRole.ANALYZER].execute_task(analysis_task)
        logger.info(f"Analysis response: {analysis_response.content}")

        synthesis_task = LLMTask(
            task_id="synthesis_001",
            description="Synthesize medical information",
            prompt=f"""Create a comprehensive response to the medical question based on the research and analysis:

                    Original Question: {question}
                    
                    Research Findings:
                    {research_response.content}
                    
                    Analysis Results:
                    {analysis_response.content}
                    
                    Create a well-structured, informative response that:
                    1. Directly addresses the original question
                    2. Presents information clearly and logically
                    3. Includes relevant details from the research
                    4. Incorporates insights from the analysis
                    5. Maintains appropriate medical disclaimers
                    6. Is accessible to a general audience""",
            system_prompt=self.agents[LLMRole.SYNTHESIZER].system_prompts[LLMRole.SYNTHESIZER],
        )
        synthesis_response = await self.agents[LLMRole.SYNTHESIZER].execute_task(synthesis_task)
        logger.info(f"Synthesis response: {synthesis_response.content}")

        validation_task = LLMTask(
            task_id="validation_001",
            description="Validate final medical response",
            prompt=f"""Review this final medical response for safety and accuracy:

                    {synthesis_response.content}
                    
                    Validate that the response:
                    1. Does not provide specific medical diagnosis or treatment advice
                    2. Includes appropriate disclaimers about consulting healthcare professionals
                    3. Presents information responsibly and safely
                    4. Does not contain misleading or potentially harmful content
                    5. Maintains educational focus
                    
                    Provide the final validated response or suggest necessary modifications.""",
            system_prompt=self.agents[LLMRole.VALIDATOR].system_prompts[LLMRole.VALIDATOR],
        )
        validation_response = await self.agents[LLMRole.VALIDATOR].execute_task(validation_task)
        logger.info(f"Validation response: {validation_response.content}")

        result = {
            "question": question,
            "search_results": search_results,
            "agent_responses": {
                "research": {
                    "content": research_response.content,
                    "provider": research_response.provider.value,
                    "model": research_response.model
                },
                "analysis": {
                    "content": analysis_response.content,
                    "provider": analysis_response.provider.value,
                    "model": analysis_response.model
                },
                "synthesis": {
                    "content": synthesis_response.content,
                    "provider": synthesis_response.provider.value,
                    "model": synthesis_response.model
                },
                "validation": {
                    "content": validation_response.content,
                    "provider": validation_response.provider.value,
                    "model": validation_response.model
                }
            },
            "final_answer": validation_response.content,
            "timestamp": asyncio.get_event_loop().time()
        }

        logger.info("Medical question processing completed")
        return result