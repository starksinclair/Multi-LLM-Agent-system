import logging
import os
from abc import ABC
from typing import Optional

from google import genai
from google.genai import types
from llm_provider import BaseLLM, LLMResponse, LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM, ABC):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set either as an argument or in the environment variables.")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        client = genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    system_instruction="You are an expert medical search query optimizer. Your goal is to transform user questions into precise and effective search queries for medical research."
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
        return LLMProvider.GEMINI