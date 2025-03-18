import io
from typing import Type, Optional, Dict
from pydantic import BaseModel

from ..base_model_toolkit import BaseModelToolkit
from ..ai_config import Model
from ..moderation import ModerationError


class AudioRecognitionModel(BaseModelToolkit):
    model_name = "audio_recognition"

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.AudioRecognition.Models

    def _prepare_input(
        self,
        in_memory_audio_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> dict:
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()
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

    def run(
        self,
        in_memory_audio_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_audio_stream, system_prompt, pydantic_model
        )
        result = self._get_structured_output(**kwargs)
        # TODO: check moderation before running the model
        if (
            self.ai_config.AudioRecognition.moderation_needed
            and not self.moderation.moderate_text(result.json())
        ):
            raise ModerationError(
                f"Audio description '{result}' was rejected by the moderation system"
            )
        return result

    async def arun(
        self,
        in_memory_audio_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_audio_stream, system_prompt, pydantic_model
        )
        result = await self._aget_structured_output(**kwargs)
        # TODO: check moderation before running the model
        if (
            self.ai_config.AudioRecognition.moderation_needed
            and not self.moderation.moderate_text(result.json())
        ):
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
