import io
from typing import Type, Optional, Dict, Any
from langchain_core.pydantic_v1 import BaseModel

from ..base_model_toolkit import BaseModelToolkit
from ..ai_config import Model
from ..prompt_manager import PromptManager
from ..moderation import ModerationError


class VisionModel(BaseModelToolkit):
    model_name = "vision"

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.Vision.Models

    def _prepare_input(
        self,
        in_memory_image_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_vision()
        # Encode in base64:
        image_base64 = VisionModel.get_b64_from_bytes(in_memory_image_stream)
        return {
            "input_dict": {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
            "system_prompt": system_prompt,
            "pydantic_model": pydantic_model,
        }

    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_image_stream, system_prompt, pydantic_model
        )
        result = self.get_structured_output(**kwargs)
        # TODO: check moderation before running the model
        if (
            self.ai_config.Vision.moderation_needed
            and not self.moderation.moderate_text(str(result))
        ):
            raise ModerationError(
                f"Image description '{result}' was rejected by the moderation system"
            )
        return result

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        kwargs = self._prepare_input(
            in_memory_image_stream, system_prompt, pydantic_model
        )
        result = await self.aget_structured_output(**kwargs)
        # TODO: check moderation before running the model
        if (
            self.ai_config.Vision.moderation_needed
            and not self.moderation.moderate_text(str(result))
        ):
            raise ModerationError(
                f"Image description '{result}' was rejected by the moderation system"
            )
        return result

    def get_price(
        self,
        image_pixels_count: int,
        *args,
        **kwargs,
    ) -> float:
        input_pixel_price = self.get_model().rate.input_pixel_price
        return image_pixels_count * input_pixel_price
