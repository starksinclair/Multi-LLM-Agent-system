import logging
from dotenv import load_dotenv
from llm_agents.multi_llm_controller import MultiLLMController, AgentResult

# mcp = FastMCP("web-search")
# URL = "https://serpapi.com/search"
from mcp_services.mcp_server.mcp_web_search_server import MCPWebSearchServer
server = MCPWebSearchServer()
controller = MultiLLMController()
load_dotenv()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPWebSearchClient:
    async def run(self, query: str) -> str |AgentResult:
        if not query:
            raise ValueError("Query cannot be empty.")

        data = await controller.process_medical_question(query)
        return data