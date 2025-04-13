import io
import os
from typing import Generator, AsyncGenerator, Optional
from pydantic import BaseModel
from .ai_config import AIConfig
from .audio_recognition_model.audio_recognition_model import (
    AudioRecognitionModel,
)
from .image_generation_model.image_generation_model import ImageGenerationModel
from .text_model.text_model import TextModel
from .vision_model.vision_model import VisionModel


class ModelsToolkit:
    def __init__(
        self, openai_api_key: Optional[str] = None, ai_config: Optional[AIConfig] = None
    ):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set in the environment! "
                    "Set it for these integration tests."
                )
        if ai_config is None:
            try:
                ai_config = AIConfig.load("ai_config.yaml")
            except FileNotFoundError:
                raise ValueError(
                    "ai_config.yaml file not found! "
                    "Set it for these integration tests."
                )
        self.text_model = TextModel(ai_config=ai_config, openai_api_key=openai_api_key)
        self.vision_model = VisionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.image_generation_model = ImageGenerationModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.audio_recognition_model = AudioRecognitionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )

    def get_text_response(self, user_input: str) -> BaseModel:
        messages = TextModel.compose_messages_openai(user_input)
        return self.text_model.run(messages)

    def get_image_description(self, in_memory_image: io.BytesIO) -> BaseModel:
        return self.vision_model.run(
            in_memory_image=in_memory_image,
        )

    def generate_image(self, prompt: str) -> BaseModel:
        return self.image_generation_model.run(
            text_description=prompt,
        )

    def get_audio_information(self, in_memory_audio_stream: io.BytesIO) -> BaseModel:
        return self.audio_recognition_model.run(
            in_memory_audio_stream=in_memory_audio_stream,
        )

    async def aget_get_text_response(self, user_input: str) -> BaseModel:
        messages = TextModel.compose_messages_openai(user_input)
        return await self.text_model.arun(messages)

    async def aget_get_image_description(
        self, in_memory_image: io.BytesIO
    ) -> BaseModel:
        return await self.vision_model.arun(
            in_memory_image=in_memory_image,
        )

    async def aget_generate_image(self, prompt: str) -> BaseModel:
        return await self.image_generation_model.arun(
            text_description=prompt,
        )

    async def aget_get_audio_information(
        self, in_memory_audio_stream: io.BytesIO
    ) -> BaseModel:
        return await self.audio_recognition_model.arun(
            in_memory_audio_stream=in_memory_audio_stream,
        )

    def stream_text_response(self, user_input: str) -> Generator[BaseModel, None, None]:
        messages = TextModel.compose_messages_openai(user_input)
        yield from self.text_model.stream(messages)

    async def astream_text_response(
        self, user_input: str
    ) -> AsyncGenerator[BaseModel, None]:
        messages = TextModel.compose_messages_openai(user_input)
        async for response in self.text_model.astream(messages=messages):
            yield response
