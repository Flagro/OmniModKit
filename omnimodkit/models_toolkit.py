from openai import OpenAI

from .ai_config import AIConfig
from .models_tools import (
    TextModel,
    VisionModel,
    ImageGenerationModel,
    AudioRecognitionModel,
)


class ModelsToolkit:
    def __init__(self, openai_api_key: str, ai_config: AIConfig):
        self.ai_config = ai_config
        self.text_model = TextModel(openai_api_key)
        self.vision_model = VisionModel(openai_api_key)
        self.image_generation_model = ImageGenerationModel(openai_api_key)
        self.audio_recognition_model = AudioRecognitionModel(openai_api_key)

    def get_price(
        self,
        token_len: int,
        audio_length: int,
        image_pixels_count: int,
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
        input_token_price = self._get_default_model("text").rate.input_token_price
        output_token_price = self._get_default_model("text").rate.output_token_price
        input_pixel_price = self._get_default_model("vision").rate.input_pixel_price
        output_pixel_price = self._get_default_model(
            "image_generation"
        ).rate.output_pixel_price

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

        return (
            token_len * input_token_price
            + token_len * output_token_price
            + audio_length * input_token_price
            + image_pixels_count * input_pixel_price
            + image_generation_pixels * output_pixel_price
        )
