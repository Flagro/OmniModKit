import io
import base64
from typing import Type, Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser

from ..base_model_toolkit import BaseModelToolkit
from ...prompt_manager import PromptManager


class VisionModel(BaseModelToolkit):
    def __init__(self, model):
        self.client = model

    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        if pydantic_object is None:
            pydantic_object = PromptManager.get_default_image_information()
        # Encode in base64:
        image_base64 = base64.b64encode(in_memory_image_stream.getvalue()).decode()
        parser = JsonOutputParser(pydantic_object=pydantic_object)

        # TODO: implement image chain runnable
        # TODO: pass the image model here

        return pydantic_object(
            image_description="an image", image_type="picture", main_objects=["image"]
        )

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        # TODO: make it non-blocking
        return self.run(in_memory_image_stream, pydantic_object)

    def get_price(
        self,
        image_pixels_count: int,
    ) -> float:
        input_pixel_price = self._get_default_model("vision").rate.input_pixel_price
        return image_pixels_count * input_pixel_price
