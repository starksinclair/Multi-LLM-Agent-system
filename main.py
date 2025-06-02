import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from mcp_services.mcp_client.search_mcp_client import MCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# Serve static files (like script.js)
app.mount("/utils", StaticFiles(directory="utils"), name="utils")

client = MCPClient()


class QueryRequest(BaseModel):
    """
       Pydantic model for validating the structure of incoming query requests.

       Attributes:
           query (str): The medical question string submitted by the user.
       """
    query: str


@app.post("/mcp")
async def run_mcp(request: QueryRequest):
    """
    API endpoint to process a medical question using the Multi-LLM pipeline.

    This asynchronous endpoint receives a user's query, passes it to the
    `MCPWebSearchClient` for comprehensive processing, and returns the
    structured result. It includes robust error handling to catch and
    report issues during the LLM operations.

    Args:
        request (QueryRequest): The incoming request body containing the user's query.

    Returns:
        AgentResult: A Pydantic object containing the question, search results,
                     responses from individual agents, and the final HTML answer.
                     FastAPI automatically serializes this object to JSON.

    Raises:
        HTTPException: If an error occurs during the processing of the query,
                       a 500 Internal Server Error is returned with the error details.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        logger.info(f"Received query: {request.query}")
        result = await client.run(request.query)
        logger.info(f"Search results: DONE")
        return result
    except Exception as error:
        logger.error(f"Error during MCP operation: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get('/', response_class=HTMLResponse)
def index():
    """
        Serves the main HTML page for the HealthConnect web application.

        This endpoint returns the `index.html` content, which includes the user interface
        for submitting medical questions and displaying answers. It incorporates a
        cache-busting timestamp for the `script.js` file to ensure the latest version is loaded.

        Returns:
            HTMLResponse: The HTML content of the main application page.
        """
    timestamp = int(time.time())
    return HTMLResponse(content=f"""
    <html>
  <head>
    <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
    <link
      rel="stylesheet"
      as="style"
      onload="this.rel='stylesheet'"
      href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
    />

     <title>MCP Web Search</title>
    <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js"></script>
  </head>
  <body>
    <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
      <div class="layout-container flex h-full grow flex-col">
        <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f0f3f4] px-10 py-3">
          <div class="flex items-center gap-4 text-[#111518]">
            <div class="size-4">
              <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M39.475 21.6262C40.358 21.4363 40.6863 21.5589 40.7581 21.5934C40.7876 21.655 40.8547 21.857 40.8082 22.3336C40.7408 23.0255 40.4502 24.0046 39.8572 25.2301C38.6799 27.6631 36.5085 30.6631 33.5858 33.5858C30.6631 36.5085 27.6632 38.6799 25.2301 39.8572C24.0046 40.4502 23.0255 40.7407 22.3336 40.8082C21.8571 40.8547 21.6551 40.7875 21.5934 40.7581C21.5589 40.6863 21.4363 40.358 21.6262 39.475C21.8562 38.4054 22.4689 36.9657 23.5038 35.2817C24.7575 33.2417 26.5497 30.9744 28.7621 28.762C30.9744 26.5497 33.2417 24.7574 35.2817 23.5037C36.9657 22.4689 38.4054 21.8562 39.475 21.6262ZM4.41189 29.2403L18.7597 43.5881C19.8813 44.7097 21.4027 44.9179 22.7217 44.7893C24.0585 44.659 25.5148 44.1631 26.9723 43.4579C29.9052 42.0387 33.2618 39.5667 36.4142 36.4142C39.5667 33.2618 42.0387 29.9052 43.4579 26.9723C44.1631 25.5148 44.659 24.0585 44.7893 22.7217C44.9179 21.4027 44.7097 19.8813 43.5881 18.7597L29.2403 4.41187C27.8527 3.02428 25.8765 3.02573 24.2861 3.36776C22.6081 3.72863 20.7334 4.58419 18.8396 5.74801C16.4978 7.18716 13.9881 9.18353 11.5858 11.5858C9.18354 13.988 7.18717 16.4978 5.74802 18.8396C4.58421 20.7334 3.72865 22.6081 3.36778 24.2861C3.02574 25.8765 3.02429 27.8527 4.41189 29.2403Z"
                  fill="currentColor"
                ></path>
              </svg>
            </div>
            <h2 class="text-[#111518] text-lg font-bold leading-tight tracking-[-0.015em]">HealthConnect</h2>
          </div>
          <div class="flex flex-1 justify-end gap-8">
            <div class="flex items-center gap-9">
              <a class="text-[#111518] text-sm font-medium leading-normal" href="http://0.0.0.0:8000/#">Home</a>
                <a class="text-[#111518] text-sm font-medium leading-normal" href="http://0.0.0.0:8000/about">About</a>
            </div>
          </div>
        </header>
        <div class="px-40 flex flex-1 justify-center py-5">
          <div class="layout-content-container flex flex-col max-w-[960px] flex-1">
            <h2 class="text-[#111518] tracking-light text-[28px] font-bold leading-tight px-4 text-center pb-3 pt-5">Ask a Medical Question</h2>
            <div class="flex max-w-[480px] flex-wrap items-end gap-4 px-4 py-3">
              <label class="flex flex-col min-w-40 flex-1">
                <textarea
                  id="question" 
                  placeholder="e.g. What causes migraines?"
                  class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#111518] focus:outline-0 focus:ring-0 border border-[#dce1e5] bg-white focus:border-[#dce1e5] min-h-36 placeholder:text-[#637988] p-[15px] text-base font-normal leading-normal"
                ></textarea>
              </label>
            </div>
            <div class="flex px-4 py-3 justify-center">
              <button
                id="askBtn"
                class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 bg-[#1993e5] text-white text-sm font-bold leading-normal tracking-[0.015em]"
              >
                <span class="truncate">Submit</span>
              </button>
            </div>
            <h2 class="text-[#111518] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Answers</h2>
            <p id="answerText" class="text-[#111518] text-base font-normal leading-normal pb-3 pt-1 px-4">Answers will appear here after you submit your question.</p>
          </div>
        </div>
      </div>
    </div>
    <script src="/utils/script.js?v={timestamp}"></script>
  </body>
</html>
    """)


@app.get("/about", response_class=HTMLResponse)
def about():
    """
        Serves the About page for the HealthConnect application.

        This endpoint provides detailed information about the project's purpose,
        technology stack, and how it leverages AI and web search to deliver
        reliable medical information.

        Returns:
            HTMLResponse: The HTML content of the About page.
        """
    return HTMLResponse(content="""
        <html>
      <head>
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
        <link
          rel="stylesheet"
          as="style"
          onload="this.rel='stylesheet'"
          href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />

        <title>Stitch Design</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
      </head>
      <body>
        <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
          <div class="layout-container flex h-full grow flex-col">
            <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f0f3f4] px-10 py-3">
              <div class="flex items-center gap-4 text-[#111518]">
                <div class="size-4">
                  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M39.475 21.6262C40.358 21.4363 40.6863 21.5589 40.7581 21.5934C40.7876 21.655 40.8547 21.857 40.8082 22.3336C40.7408 23.0255 40.4502 24.0046 39.8572 25.2301C38.6799 27.6631 36.5085 30.6631 33.5858 33.5858C30.6631 36.5085 27.6632 38.6799 25.2301 39.8572C24.0046 40.4502 23.0255 40.7407 22.3336 40.8082C21.8571 40.8547 21.6551 40.7875 21.5934 40.7581C21.5589 40.6863 21.4363 40.358 21.6262 39.475C21.8562 38.4054 22.4689 36.9657 23.5038 35.2817C24.7575 33.2417 26.5497 30.9744 28.7621 28.762C30.9744 26.5497 33.2417 24.7574 35.2817 23.5037C36.9657 22.4689 38.4054 21.8562 39.475 21.6262ZM4.41189 29.2403L18.7597 43.5881C19.8813 44.7097 21.4027 44.9179 22.7217 44.7893C24.0585 44.659 25.5148 44.1631 26.9723 43.4579C29.9052 42.0387 33.2618 39.5667 36.4142 36.4142C39.5667 33.2618 42.0387 29.9052 43.4579 26.9723C44.1631 25.5148 44.659 24.0585 44.7893 22.7217C44.9179 21.4027 44.7097 19.8813 43.5881 18.7597L29.2403 4.41187C27.8527 3.02428 25.8765 3.02573 24.2861 3.36776C22.6081 3.72863 20.7334 4.58419 18.8396 5.74801C16.4978 7.18716 13.9881 9.18353 11.5858 11.5858C9.18354 13.988 7.18717 16.4978 5.74802 18.8396C4.58421 20.7334 3.72865 22.6081 3.36778 24.2861C3.02574 25.8765 3.02429 27.8527 4.41189 29.2403Z"
                      fill="currentColor"
                    ></path>
                  </svg>
                </div>
                <h2 class="text-[#111518] text-lg font-bold leading-tight tracking-[-0.015em]">HealthConnect</h2>
              </div>
              <div class="flex flex-1 justify-end gap-8">
                <div class="flex items-center gap-9">
                  <a class="text-[#111518] text-sm font-medium leading-normal" href="http://0.0.0.0:8000/#">Home</a>
                    <a class="text-[#111518] text-sm font-medium leading-normal" href="http://0.0.0.0:8000/about">About</a>
                </div>
              </div>
            </header>
           <div class="layout-content-container flex flex-col max-w-[960px] mx-auto px-6 py-10 space-y-6 text-[#111518]">
            <h2 class="text-3xl font-bold text-center">About HealthConnect</h2>
            
            <p class="text-base leading-relaxed">
            <strong>HealthConnect</strong> is a cutting-edge web application designed to provide users with accurate, reliable, and validated answers to their medical questions. Our mission is to empower individuals with accessible health information, leveraging the power of advanced AI and comprehensive web search.
            </p>
            
            <p class="text-base leading-relaxed">
            At its core, HealthConnect employs a sophisticated <strong>multi-agent LLM pipeline</strong>. This means your queries are not handled by a single AI, but by a specialized team of Large Language Models, each with a distinct role:
            </p>
            
            <ul class="list-disc pl-6 space-y-2 text-base leading-relaxed">
            <li><strong>Query Refiner:</strong> Optimizes your original question into a precise search query for maximum relevance.</li>
            <li><strong>Researcher:</strong> Conducts extensive web searches, focusing on reputable medical sources like Mayo Clinic, WebMD, NIH, and peer-reviewed journals.</li>
            <li><strong>Validator:</strong> Critically reviews the research findings to ensure accuracy, safety, and adherence to medical guidelines, adding crucial disclaimers.</li>
            </ul>
            
            <p class="text-base leading-relaxed">
            This multi-layered approach ensures that the information you receive is not only comprehensive but also carefully vetted for safety and educational value. HealthConnect is built with <strong>FastAPI</strong> for a robust backend and a responsive, user-friendly frontend, providing real-time answers to your health inquiries.
            </p>
            
            <div class="bg-yellow-100 border-l-4 border-yellow-400 p-4 text-sm text-yellow-900 rounded-md">
            <strong>Disclaimer:</strong> The information provided by HealthConnect is for educational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
            </div>
            </div>
          </div>
        </div>
      </body>
    </html>
        """)
