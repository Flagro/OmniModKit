import io
from typing import Type, Optional, Dict
from langchain_core.pydantic_v1 import BaseModel

from ..base_model_toolkit import BaseModelToolkit
from ..ai_config import Model
from ..prompt_manager import PromptManager


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
            pydantic_model = PromptManager.get_default_audio_information()
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_audio()
        # Encode in base64:
        audio_base64 = AudioRecognitionModel.get_b64_from_bytes(in_memory_audio_stream)
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
        return self.get_structured_output(**kwargs)

    async def arun(
        self,
        in_memory_audio_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_audio_stream, system_prompt, pydantic_model
        )
        return await self.aget_structured_output(**kwargs)

    def get_price(
        self,
        audio_length: int,
        *args,
        **kwargs,
    ) -> float:
        input_audio_second_price = self.get_model().rate.input_audio_second_price
        return audio_length * input_audio_second_price
