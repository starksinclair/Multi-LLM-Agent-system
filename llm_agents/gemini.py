import logging
import os
from abc import ABC
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .llm_provider import BaseLLM, LLMResponse, LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class GeminiLLM(BaseLLM, ABC):
    """
    GeminiLLM class provides an interface to interact with Google's Gemini Large Language Models (LLMs).

    This class extends `BaseLLM` and implements methods for generating text responses
    and identifying the LLM provider. It handles API key validation and basic error logging
    specific to the Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set either as an argument or in the environment variables.")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generates a text response from the Gemini LLM based on the given prompt and system prompt.

        This method initializes a `google.genai.Client` and then calls the `generate_content`
        method with the specified model, configuration (temperature, max output tokens,
        and system instruction), and user content. It wraps the response in an `LLMResponse`
        object and includes error handling for API failures.

        Args:
            prompt (str): The user's input prompt for the LLM.
            system_prompt (Optional[str]): An optional system-level instruction or context
                                           to guide the LLM's behavior. Defaults to None.

        Returns:
            LLMResponse: An object containing the generated content, the provider (GEMINI),
                         and the model used. In case of an error, it will contain an error
                         message in the content field.

        """
        client = genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    system_instruction=system_prompt
                ),
                contents=prompt
            )
            return LLMResponse(
                content=response.text.strip(),
                provider=LLMProvider.GEMINI,
                model=self.model
            )
        except Exception as error:
            logger.error(f"OpenAI API error: {error}")
            return LLMResponse(
                content=f"Error generating response: {error}",
                provider=LLMProvider.GEMINI,
                model="gemini-model"
            )

    def get_provider(self) -> LLMProvider:
        """
       Returns the LLM provider for this instance.

       Returns:
           LLMProvider: An enum member indicating the LLM provider, which is LLMProvider.GEMINI.
        """
        return LLMProvider.GEMINI
