import io
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from openai import AsyncOpenAI

from ..base_toolkit_model import BaseToolkitModel, OpenAIMessage
from ..ai_config import AudioRecognition
from ..moderation import ModerationError


class DefaultAudioInformation(BaseModel):
    audio_description: str = Field(description="a short description of the audio")

    def __str__(self):
        return self.audio_description


class AudioRecognitionModel(BaseToolkitModel[DefaultAudioInformation]):
    model_name = "audio_recognition"
    default_pydantic_model: Type[DefaultAudioInformation] = DefaultAudioInformation

    @staticmethod
    def get_default_system_prompt() -> str:
        return "Based on the audio, fill out the provided fields."

    def get_model_config(self) -> AudioRecognition:
        return self.ai_config.audio_recognition

    def run_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        default_pydantic_model = self.default_pydantic_model
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}"
            )
        client = OpenAI(api_key=self.openai_api_key)
        in_memory_audio_stream.seek(0)
        transcript = client.audio.transcriptions.create(
            file=in_memory_audio_stream,
            model=self.get_model().name,
        )
        result = default_pydantic_model(
            audio_description=transcript.text,
        )
        # TODO: check moderation before running the model
        if self.moderation_needed and not self.moderate_text(result.model_dump_json()):
            raise ModerationError(
                f"Audio description '{result}' was rejected by the moderation system"
            )
        return result

    async def arun_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        default_pydantic_model = self.default_pydantic_model
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}"
            )
        client = AsyncOpenAI(api_key=self.openai_api_key)
        in_memory_audio_stream.seek(0)
        transcript = await client.audio.transcriptions.create(
            file=in_memory_audio_stream,
            model=self.get_model().name,
        )
        result = self.default_pydantic_model(
            audio_description=transcript.text,
        )
        # TODO: check moderation before running the model
        if self.moderation_needed and not self.moderate_text(result.model_dump_json()):
            raise ModerationError(
                f"Audio description '{result}' was rejected by the moderation system"
            )
        return result

    def get_price(
        self,
        input_audio: Optional[io.BytesIO] = None,
        output_text: Optional[str] = None,
    ) -> float:
        price = 0.0
        if input_audio is not None:
            audio_length = 100  # Placeholder for audio length in seconds
            input_audio_second_price = self.get_model().rate.input_audio_second_price
            price += audio_length * input_audio_second_price
        if output_text is not None:
            output_token_length = self.count_tokens(output_text)
            output_token_price = self.get_model().rate.output_token_price
            price += output_token_length * output_token_price
        return price
