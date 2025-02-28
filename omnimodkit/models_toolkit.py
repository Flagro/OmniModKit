import io
from typing import Optional, Generator, AsyncGenerator
from pydantic import BaseModel
from .ai_config import AIConfig
from .audio_recognition_model.audio_recognition_model import (
    AudioRecognitionModel,
)
from .image_generation_model.image_generation_model import ImageGenerationModel
from .text_model.text_model import TextModel
from .vision_model.vision_model import VisionModel
from .prompt_manager import PromptManager


class UniversalModelsToolkit:
    def __init__(self, openai_api_key: str, ai_config: AIConfig):
        self.text_model = TextModel(openai_api_key, ai_config)
        self.vision_model = VisionModel(openai_api_key, ai_config)
        self.image_generation_model = ImageGenerationModel(openai_api_key, ai_config)
        self.audio_recognition_model = AudioRecognitionModel(openai_api_key, ai_config)

    def get_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> BaseModel:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return self.models_toolkit.run_model("text", messages)

    def get_image_description(self, in_memory_image: io.BytesIO) -> BaseModel:
        return self.models_toolkit.run_model(
            "vision",
            in_memory_image=in_memory_image,
            system_prompt=PromptManager.get_default_system_prompt_vision(),
        )

    def generate_image(self, prompt: str) -> BaseModel:
        return self.models_toolkit.run_model(
            "image_generation",
            text_description=prompt,
            system_prompt=PromptManager.get_default_system_prompt_image(),
        )

    def get_audio_information(self, in_memory_audio_stream: io.BytesIO) -> BaseModel:
        return self.models_toolkit.run_model(
            "audio_recognition",
            in_memory_audio_stream=in_memory_audio_stream,
            system_prompt=PromptManager.get_default_system_prompt_audio(),
        )

    async def agent_get_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> BaseModel:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return await self.models_toolkit.arun_model("text", messages=messages)

    async def agent_get_image_description(
        self, in_memory_image: io.BytesIO
    ) -> BaseModel:
        return await self.models_toolkit.arun_model(
            "vision",
            in_memory_image=in_memory_image,
            system_prompt=PromptManager.get_default_system_prompt_vision(),
        )

    async def agent_generate_image(self, prompt: str) -> BaseModel:
        return await self.models_toolkit.arun_model(
            "image_generation",
            text_description=prompt,
            system_prompt=PromptManager.get_default_system_prompt_image(),
        )

    async def agent_get_audio_information(
        self, in_memory_audio_stream: io.BytesIO
    ) -> BaseModel:
        return await self.models_toolkit.arun_model(
            "audio_recognition",
            in_memory_audio_stream=in_memory_audio_stream,
            system_prompt=PromptManager.get_default_system_prompt_audio(),
        )

    def stream_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> Generator[BaseModel]:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return self.models_toolkit.stream_model("text", messages=messages)

    async def astream_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> AsyncGenerator[BaseModel]:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        async for response in self.models_toolkit.astream_model(
            "text", messages=messages
        ):
            yield response
