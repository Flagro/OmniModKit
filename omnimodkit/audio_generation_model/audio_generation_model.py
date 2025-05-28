import io
from typing import Type, List
from pydantic import BaseModel
from openai import OpenAI
from openai import AsyncOpenAI

from ..base_toolkit_model import BaseToolkitModel, OpenAIMessage
from ..ai_config import AudioGeneration
from ..moderation import ModerationError


class AudioGenerationModel(BaseToolkitModel):
    model_name = "audio_generation"

    def get_model_config(self) -> AudioGeneration:
        return self.ai_config.audio_generation

    def run_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        default_pydantic_model = self.get_default_pydantic_model()
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}"
            )
        client = OpenAI(api_key=self.openai_api_key)
        raise NotImplementedError(
            "Audio generation is not implemented yet. Please implement the audio generation logic."
        )

    async def arun_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        default_pydantic_model = self.get_default_pydantic_model()
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}"
            )
        client = AsyncOpenAI(api_key=self.openai_api_key)
        raise NotImplementedError(
            "Audio generation is not implemented yet. Please implement the audio generation logic."
        )

    def get_price(
        self,
        audio_length: int,
        *args,
        **kwargs,
    ) -> float:
        input_audio_second_price = self.get_model().rate.input_audio_second_price
        return audio_length * input_audio_second_price
