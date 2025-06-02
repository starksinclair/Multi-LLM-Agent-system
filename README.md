# HealthConnect: Multi-Agent Medical Question Answering System

HealthConnect is a web application that delivers accurate, reliable, and medically validated answers to user questions by leveraging a multi-agent Large Language Model (LLM). It integrates with Model Context Protocol (MCP) servers to perform comprehensive web and medical literature searches, ensuring responses are grounded in up-to-date and trustworthy information.

## 🧠 Architecture and How It Works

HealthConnect operates through a multi-stage pipeline orchestrated by several key components:

1.  **`MultiLLMController`**: The central orchestrator that manages the entire question-answering workflow. It initializes and coordinates various LLM agents.

2.  **`MedicalLLMController`**: Manages the interaction with a specific LLM instance for a defined role (e.g., `QUERY_REFINER`, `RESEARCHER`, `VALIDATOR`). It applies role-specific system prompts and integrates search context.

3.  **`MCPClient`**: Acts as the client interface for external components to submit medical queries. It connects to the MCP server, refines the initial query, performs web and PubMed searches, and then passes the results to the `MultiLLMController`.

4.  **`WebSearchHelper`**: A utility class that interacts with SerpAPI to perform general web searches and format the results for LLM consumption.

5.  **`PubMedHelper`**: A utility class that interacts with the NCBI E-utilities API to search for medical literature on PubMed and fetch article abstracts.

6.  **FastAPI Server**: Provides the web interface for the application, handling incoming user queries via an API endpoint (`/mcp`) and serving the main HTML application (`/`) and an About page (`/about`).

7.  **MCP Server (`mcp_server/search.py`)**: An MCP (Multi-Component Protocol) server that exposes `search_pubmed` and `web_search` as callable tools, allowing the `MCPClient` to request search operations.

When a user submits a medical question:

* The FastAPI backend receives the query.

* The `MCPClient` refines the query for optimal search.

* The `MCPClient` calls the `search_pubmed` and `web_search` tools on the MCP server to get relevant information.

* The `MultiLLMController` then uses a `RESEARCHER` LLM to synthesize the information from both search results.

* Finally, a `VALIDATOR` LLM reviews the synthesized response, ensuring accuracy, safety, and proper disclaimers, and formats it into HTML.

* The final HTML answer is returned to the user's browser.

## 🚀 Getting Started

To get HealthConnect up and running on your local machine, follow these steps:

### Prerequisites

* [Docker](https://www.docker.com/get-started/) installed on your system.

### Environment Variables

Before running the application, you need to create a `.env` file in the root directory of the project and populate it with your API keys.

Create a file named `.env` with the following content:
```
SERPAPI_KEY="your_serpapi_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"

```

**Note:** Replace `"your_api_key_here"` with your actual API keys obtained from the respective providers.

### Installation and Running the Application

1.  **Clone the repository:**

    ```
    git clone https://github.com/starksinclair/Multi-LLM-Agent-system.git
    cd HealthConnect

    ```

2.  **Build the Docker image and  run the Docker container:**

    ```
    docker compose up --build

    ```

    This command will use the `Dockerfile` and `requirements.txt` to build the necessary image, installing all Python dependencies and maps port 8000 from the container to port 8000 on your host, and injects the environment variables from your `.env` file into the container.

3.  **Access the application:**
    Open your web browser and navigate to:

    ```
    http://localhost:8000

    ```

## 🌐 API Endpoints

* **`/` (GET)**: Serves the main HTML page of the HealthConnect web application, where users can submit medical questions.

* **`/about` (GET)**: Serves the About page, providing information about the project.

* **`/mcp` (POST)**:

    * **Method**: `POST`

    * **Request Body**: `{"query": "Your medical question here"}`

    * **Description**: Processes a medical question through the multi-LLM pipeline and returns a structured `AgentResult` object containing the final answer and intermediate agent responses.

## 📚 Resources Used

* **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python.

* **Pydantic**: Data validation and settings management using Python type hints.

* **OpenAI API**: For interacting with OpenAI's Large Language Models.

* **Google Gemini API**: For interacting with Google's Gemini Large Language Models.

* **DeepSeek API**: For interacting with DeepSeek's Large Language Models.

* **SerpAPI**: A real-time API to access Google search results.

* **PubMed/NCBI E-utilities API**: For programmatic access to PubMed's biomedical literature database.

* **Multi-Component Protocol (MCP)**: A framework for building modular and extensible AI systems.

* **Docker**: For containerizing the application, ensuring consistent environments.

* **`python-dotenv`**: For loading environment variables from a `.env` file.

* **`requests`**: For making HTTP requests to external APIs.

## ⚠️ Disclaimer

The information provided by HealthConnect is for educational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
