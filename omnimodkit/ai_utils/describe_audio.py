import io
from typing import Type
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from .base_utility import BaseUtility


class DefaultAudioInformation(BaseModel):
    # TODO: move this to prompt manager
    audio_description: str = Field(description="a short description of the audio")

    def __str__(self):
        return self.audio_description


class DescribeAudioUtililty(BaseUtility):
    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Type[BaseModel] = DefaultAudioInformation,
    ) -> BaseModel:
        # Encode in base64:
        parser = JsonOutputParser(pydantic_object=pydantic_object)

        return pydantic_object(audio_description="an audio")

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Type[BaseModel] = DefaultAudioInformation,
    ) -> BaseModel:
        # TODO: make it non-blocking
        return self.run(in_memory_image_stream)
