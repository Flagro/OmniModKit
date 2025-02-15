from typing import Dict
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from ..base_model_toolkit import BaseModelToolkit
from ...prompt_manager import PromptManager
from ...ai_config import Model


class ImageGenerationModel(BaseModelToolkit):
    model_name = "image_generation"

    def run(
        self,
        text_description: str,
        system_prompt: str,
    ) -> BaseModel:
        pydantic_model = PromptManager.get_default_image()
        llm = self.get_model_chain()
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template=system_prompt + " {image_desc}",
        )
        chain = prompt | llm
        image_url = DallEAPIWrapper(api_key=self.openai_api_key).run(
            chain.invoke(text_description)
        )
        return pydantic_model(image_url=image_url)

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.ImageGeneration.Models

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

        image_generation_dimensions = self.ai_config.ImageGeneration.output_image_size
        if "x" not in image_generation_dimensions:
            raise ValueError(
                f"Invalid image generation dimensions: {image_generation_dimensions}"
            )
        image_generation_dimensions_x, image_generation_dimensions_y = (
            image_generation_dimensions.split("x")
        )
        image_generation_pixels = (
            0
            if not image_generation_needed
            else int(image_generation_dimensions_x) * int(image_generation_dimensions_y)
        )

        return image_generation_pixels * output_pixel_price
