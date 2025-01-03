from typing import Dict
from openai import OpenAI
from ..base_model_toolkit import BaseModelToolkit
from ...ai_config import Model


class ImageGenerationModel(BaseModelToolkit):
    model_name = "image_generation"
    default_attribute = "image_generation_default"

    def __init__(self, openai_api_key: str):
        # TODO: fix this - this is not OpenAI object
        self.image_generation_model = OpenAI(
            api_key=openai_api_key,
            model=self.get_model().name,
        )

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.ImageGeneration.Models

    def get_price(
        self,
        image_generation_needed: bool,
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
