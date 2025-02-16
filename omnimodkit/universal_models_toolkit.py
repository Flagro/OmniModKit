import io
from typing import Optional, Generator, AsyncGenerator
from pydantic import BaseModel
from .ai_config import AIConfig
from .models_tools import (
    TextModel,
    VisionModel,
    ImageGenerationModel,
    AudioRecognitionModel,
)
from .models_toolkit import ModelsToolkit
from .prompt_manager import PromptManager


class UniversalModelsToolkit:
    def __init__(self, openai_api_key: str, ai_config: AIConfig):
        self.models_toolkit = ModelsToolkit(
            openai_api_key,
            ai_config,
            [
                TextModel,
                VisionModel,
                ImageGenerationModel,
                AudioRecognitionModel,
            ],
        )

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
            in_memory_image,
            system_prompt=PromptManager.get_default_system_prompt_vision(),
        )

    def generate_image(self, prompt: str) -> BaseModel:
        return self.models_toolkit.run_model(
            "image_generation",
            prompt,
            system_prompt=PromptManager.get_default_system_prompt_image(),
        )

    def get_audio_information(self, in_memory_audio_stream: io.BytesIO) -> BaseModel:
        return self.models_toolkit.run_model(
            "audio_recognition",
            PromptManager.get_default_system_prompt_audio(),
            in_memory_audio_stream,
        )

    async def agent_get_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> BaseModel:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return await self.models_toolkit.arun_model("text", messages)

    async def agent_get_image_description(
        self, in_memory_image: io.BytesIO
    ) -> BaseModel:
        return await self.models_toolkit.arun_model("vision", in_memory_image)

    async def agent_generate_image(self, prompt: str) -> BaseModel:
        return await self.models_toolkit.arun_model("image_generation", prompt)

    async def agent_get_audio_information(
        self, in_memory_audio_stream: io.BytesIO
    ) -> BaseModel:
        return await self.models_toolkit.arun_model(
            "audio_recognition", in_memory_audio_stream
        )

    def stream_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> Generator[BaseModel]:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return self.models_toolkit.stream_model("text", messages)

    async def astream_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> AsyncGenerator[BaseModel]:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        async for response in self.models_toolkit.astream_model("text", messages):
            yield response
