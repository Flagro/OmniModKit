import io
from typing import Type
from pydantic import BaseModel

from ..base_model import BaseModel
from ..ai_config import GenerationType
from ..moderation import ModerationError


class AudioRecognitionModel(BaseModel):
    model_name = "audio_recognition"

    def get_model_config(self) -> GenerationType:
        return self.ai_config.audio_recognition

    def _prepare_input(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        in_memory_audio_stream: io.BytesIO,
    ) -> dict:
        # Encode in base64:
        audio_base64 = self.get_b64_from_bytes(in_memory_audio_stream)
        return {
            "input_dict": {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "mp3",
                },
            },
            "system_prompt": system_prompt,
            "pydantic_model": pydantic_model,
        }

    def run_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_audio_stream, system_prompt, pydantic_model
        )
        result = self._get_structured_output(**kwargs)
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
        in_memory_audio_stream: io.BytesIO,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_audio_stream, system_prompt, pydantic_model
        )
        result = await self._aget_structured_output(**kwargs)
        # TODO: check moderation before running the model
        if self.moderation_needed and not self.moderate_text(result.model_dump_json()):
            raise ModerationError(
                f"Audio description '{result}' was rejected by the moderation system"
            )
        return result

    def get_price(
        self,
        audio_length: int,
        *args,
        **kwargs,
    ) -> float:
        input_audio_second_price = self.get_model().rate.input_audio_second_price
        return audio_length * input_audio_second_price
