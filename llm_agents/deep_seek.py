import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from .llm_provider import BaseLLM, LLMResponse, LLMProvider
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """
   DeepSeekLLM class provides an interface to interact with DeepSeek's Large Language Models (LLMs).

   This class extends `BaseLLM` and implements methods for generating text responses
   and identifying the LLM provider. It uses the OpenAI client library configured
   to connect to the DeepSeek API endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-reasoner"):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generates a text response from the DeepSeek LLM based on the given prompt and system prompt.

        This method constructs a list of messages for the DeepSeek API (via the OpenAI client),
        including an optional system message and the user's prompt. It then calls the
        `chat.completions.create` method to get a response and encapsulates it within an
        `LLMResponse` object. Basic error handling is included.

        Args:
            prompt (str): The user's input prompt for the LLM.
            system_prompt (Optional[str]): An optional system-level instruction or context
                                            to guide the LLM's behavior. Defaults to None.

        Returns:
            LLMResponse: An object containing the generated content, the provider (DEEPSEEK),
                         and the model used. In case of an error, it will contain an error
                         message in the content field.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )

            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                provider=LLMProvider.DEEPSEEK,
                model=self.model
            )
        except Exception as error:
            logger.error(f"Deep Seeker API error: {error}")
            return LLMResponse(
                content=f"Error generating response: {error}",
                provider=LLMProvider.DEEPSEEK,
                model=self.model
            )

    def get_provider(self) -> LLMProvider:
        """
        Returns the LLM provider for this instance.

        Returns:
            LLMProvider: An enum member indicating the LLM provider, which is LLMProvider.DEEPSEEK.
        """
        return LLMProvider.DEEPSEEK
