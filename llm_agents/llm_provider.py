from abc import abstractmethod, ABC
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class LLMProvider(Enum):
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

class LLMResponse(BaseModel):
    content: str
    provider: LLMProvider
    model: str


class BaseLLM(ABC):
    """
    Base class for LLM providers.
    """

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the LLM based on the provided prompt.

        Args: prompt (str): The input prompt for the LLM.

        Returns: LLMResponse: The generated response from the LLM.
        :param prompt:
        :param system_prompt:
        """
        pass

    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """
        Get the provider of the LLM.

        Returns:
            LLMProvider: The provider of the LLM.
        """
        pass
