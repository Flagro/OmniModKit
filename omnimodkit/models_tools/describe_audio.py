import io
from typing import Type, Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser

from .base_utility import BaseUtility
from ..prompt_manager import PromptManager


class DescribeAudioUtililty(BaseUtility):
    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        if pydantic_object is None:
            # TODO: prompt manager should be available as part of the utility
            pydantic_object = PromptManager.get_default_audio_information()
        # Encode in base64:
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        return pydantic_object(audio_description="an audio")

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        # TODO: make it non-blocking
        return self.run(in_memory_image_stream, pydantic_object)
