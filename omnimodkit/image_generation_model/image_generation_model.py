from typing import Type
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from ..base_toolkit_model import BaseToolkitModel
from ..ai_config import ImageGeneration
from ..moderation import ModerationError


class ImageGenerationModel(BaseToolkitModel):
    model_name = "image_generation"

    def run_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not self.moderate_text(user_input):
            raise ModerationError(
                f"Text description '{user_input}' was rejected by the moderation system"
            )
        default_pydantic_model = self.get_default_pydantic_model()
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}, "
            )
        # TODO: move client creation to __init__ method
        client = OpenAI(api_key=self.openai_api_key)
        generation_response = client.images.generate(
            model=self.get_model().name,
            prompt=f"{system_prompt}\n{user_input}",
            n=1,
            size=self.get_model_config().output_image_size,
            response_format="url",
        )
        image_url = generation_response.data[0].url
        return pydantic_model(image_url=image_url)

    async def arun_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not await self.amoderate_text(user_input):
            raise ModerationError(
                f"Text description '{user_input}' was rejected by the moderation system"
            )
        default_pydantic_model = self.get_default_pydantic_model()
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}, "
            )
        # TODO: move client creation to __init__ method
        client = AsyncOpenAI(api_key=self.openai_api_key)
        generation_response = await client.images.generate(
            model=self.get_model().name,
            prompt=f"{system_prompt}\n{user_input}",
            n=1,
            size=self.get_model_config().output_image_size,
            response_format="url",
        )
        image_url = generation_response.data[0].url
        return pydantic_model(image_url=image_url)

    def get_model_config(self) -> ImageGeneration:
        return self.ai_config.image_generation

    def get_price(
        self,
        *args,
        **kwargs,
    ) -> float:
        """
        Returns the price of the AI services for the given
        input parameters
        """
        output_pixel_price = self.get_model().rate.output_pixel_price

        image_generation_dimensions = self.get_model_config().output_image_size
        if "x" not in image_generation_dimensions:
            raise ValueError(
                f"Invalid image generation dimensions: {image_generation_dimensions}"
            )
        image_generation_dimensions_x, image_generation_dimensions_y = map(
            int, image_generation_dimensions.split("x")
        )
        total_pixels = image_generation_dimensions_x * image_generation_dimensions_y
        return total_pixels * output_pixel_price
