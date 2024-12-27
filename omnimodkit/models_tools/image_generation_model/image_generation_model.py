from ..base_model_toolkit import BaseModelToolkit


class ImageGenerationModel(BaseModelToolkit):
    def __init__(self, model):
        self.client = model

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

        return image_generation_pixels * output_pixel_price
