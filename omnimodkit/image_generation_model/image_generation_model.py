from typing import Type
from pydantic import BaseModel
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from ..base_model import BaseModel
from ..ai_config import GenerationType
from ..moderation import ModerationError


class ImageGenerationModel(BaseModel):
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
        if pydantic_model is not self.get_default_pydantic_model():
            raise ValueError(
                f"Image generation requires pydantic_model must be {self.get_default_pydantic_model()}"
            )
        llm = self.get_model_chain()
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template=system_prompt,
        )
        chain = prompt | llm
        # TODO: pass models parameters in here
        image_url = DallEAPIWrapper(api_key=self.openai_api_key).run(
            chain.invoke(user_input)
        )
        return pydantic_model(image_url=image_url)

    def get_model_config(self) -> GenerationType:
        return self.ai_config.image_generation

    def get_price(
        self,
        image_generation_needed: bool,
        *args,
        **kwargs,
    ) -> float:
        """
        Returns the price of the AI services for the given
        input parameters

        Args:
        token_len: the number of tokens in the input text
        audio_length: the length of the audio in seconds
        image_pixels_count: the number of pixels in the image
        image_generation_needed: whether the image generation is needed
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
        return total_pixels * output_pixel_price if image_generation_needed else 0
