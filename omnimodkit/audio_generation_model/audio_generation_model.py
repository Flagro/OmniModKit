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
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not self.moderate_text(user_input):
            raise ModerationError(
                f"Audio description '{user_input}' was rejected by the moderation system"
            )
        default_pydantic_model = self.get_default_pydantic_model()
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}"
            )
        client = OpenAI(api_key=self.openai_api_key)
        resp = client.audio.speech.create(
            model=self.get_model().name,
            voice=self.get_model_config().voice,
            input=user_input,
            format="ogg_opus",
            stream=True,
        )
        bytes_response = b"".join(chunk for chunk in resp.iter_bytes())
        io_bytes = io.BytesIO(bytes_response)
        io_bytes.name = "audio.ogg"
        io_bytes.seek(0)
        return pydantic_model(audio_bytes=io_bytes)

    async def arun_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not await self.amoderate_text(user_input):
            raise ModerationError(
                f"Audio description '{user_input}' was rejected by the moderation system"
            )
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
        input: str,
        *args,
        **kwargs,
    ) -> float:
        input_token_price = self.get_model().rate.input_token_price
        return self.count_tokens(input) * input_token_price
