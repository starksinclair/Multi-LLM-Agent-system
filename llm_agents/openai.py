import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from .llm_provider import BaseLLM, LLMResponse, LLMProvider
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAILLM class provides an interface to interact with OpenAI's Large Language Models (LLMs).

    This class extends `BaseLLM` and implements methods for generating text responses
    and identifying the LLM provider. It handles API key management and basic error logging.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
       Generates a text response from the OpenAI LLM based on the given prompt and system prompt.

       This method constructs a list of messages for the OpenAI API, including an optional
       system message and the user's prompt. It then calls the OpenAI API to get a response
       and encapsulates it within an LLMResponse object. Basic error handling is included.

       Args:
           prompt (str): The user's input prompt for the LLM.
           system_prompt (Optional[str]): An optional system-level instruction or context
                                           to guide the LLM's behavior. Defaults to None.

       Returns:
           LLMResponse: An object containing the generated content, the provider (OPENAI),
                        and the model used. In case of an error, it will contain an error
                        message in the content field.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=0.3,
                max_output_tokens=1000
                # max_tokens=1000,
                # temperature=1.0
            )

            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                provider=LLMProvider.OPENAI,
                model=self.model
            )
        except Exception as error:
            logger.error(f"OpenAI API error: {error}")
            return LLMResponse(
                content=f"Error generating response: {error}",
                provider=LLMProvider.OPENAI,
                model=self.model
            )

    def get_provider(self) -> LLMProvider:
        """
        Returns the LLM provider for this instance.

        Returns:
            LLMProvider: An enum member indicating the LLM provider, which is LLMProvider.OPENAI.
        """
        return LLMProvider.OPENAI
