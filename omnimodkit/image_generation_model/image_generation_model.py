from typing import Type, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from ..base_toolkit_model import BaseToolkitModel, OpenAIMessage
from ..ai_config import ImageGeneration
from ..moderation import ModerationError


class DefaultImage(BaseModel):
    image_url: str = Field(description="url of the image")

    def __str__(self):
        return f"Image url: {self.image_url}"


class ImageGenerationModel(BaseToolkitModel[DefaultImage]):
    model_name = "image_generation"
    default_pydantic_model: Type[DefaultImage] = DefaultImage

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(api_key=self.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.openai_api_key)

    @staticmethod
    def get_default_system_prompt() -> str:
        # TODO: this should not be a formatted string
        return "Please provide the necessary information: {image_desc}"

    def run_impl(
        self,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: List[OpenAIMessage],
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not self.moderate_text(user_input):
            raise ModerationError(
                f"Text description '{user_input}' was rejected by the moderation system"
            )
        default_pydantic_model = self.default_pydantic_model
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}, "
            )
        communication_history_prompt = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in communication_history]
        )
        generation_response = self.client.images.generate(
            model=self.get_model().name,
            prompt=f"{system_prompt}\nCommunication history:\n"
            f"{communication_history_prompt}\nUser input: {user_input}",
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
        communication_history: List[OpenAIMessage],
        user_input: str,
    ) -> BaseModel:
        if self.moderation_needed and not await self.amoderate_text(user_input):
            raise ModerationError(
                f"Text description '{user_input}' was rejected by the moderation system"
            )
        default_pydantic_model = self.default_pydantic_model
        if pydantic_model is not default_pydantic_model:
            raise ValueError(
                f"Image generation requires pydantic_model must be {default_pydantic_model}, "
            )
        communication_history_prompt = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in communication_history]
        )
        generation_response = await self.async_client.images.generate(
            model=self.get_model().name,
            prompt=f"{system_prompt}\nCommunication history:\n"
            f"{communication_history_prompt}\nUser input: {user_input}",
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
        input_text: Optional[str] = None,
        output_image_url: Optional[str] = None,
    ) -> float:
        """
        Returns the price of the AI services for the given
        input parameters
        """
        price = 0.0
        if output_image_url:
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
            price += total_pixels * output_pixel_price
        if input_text:
            token_cnt = self.count_tokens(input_text)
            price += self.get_model().rate.input_token_price * token_cnt
        return price
