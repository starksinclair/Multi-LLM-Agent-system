import os
from typing import Optional

from openai import OpenAI

from llm_provider import BaseLLM, LLMResponse, LLMProvider
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekerLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("DEE_SEEKER_API_KEY"), base_url="https://api.deepseeker.com")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=1.0
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
        return LLMProvider.DEEPSEEK