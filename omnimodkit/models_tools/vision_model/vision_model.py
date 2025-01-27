import io
import base64
from typing import Type, Optional, Dict, Any
from langchain_core.pydantic_v1 import BaseModel

from ..base_model_toolkit import BaseModelToolkit
from ...ai_config import Model
from ...prompt_manager import PromptManager


class VisionModel(BaseModelToolkit):
    model_name = "vision"

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.TextGeneration.Models

    def _prepare_input(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        if pydantic_model is None:
            pydantic_model = PromptManager.get_default_image_information()
        # Encode in base64:
        image_base64 = base64.b64encode(in_memory_image_stream.getvalue()).decode()
        return {
            "input_dict": {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
            "system_prompt": "Based on the image, fill out the provided fields.",
            "pydantic_model": pydantic_model,
        }

    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(in_memory_image_stream, pydantic_model)
        return self.get_structured_output(**kwargs)

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(in_memory_image_stream, pydantic_model)
        return await self.aget_structured_output(**kwargs)

    def get_price(
        self,
        image_pixels_count: int,
        *args,
        **kwargs,
    ) -> float:
        input_pixel_price = self.get_model().rate.input_pixel_price
        return image_pixels_count * input_pixel_price
