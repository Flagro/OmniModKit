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
        # TODO: pass values as kwargs
        return (
            self.audio_recognition_model.get_price(audio_length)
            + self.image_generation_model.get_price(image_generation_needed)
            + self.text_model.get_price(token_len)
            + self.vision_model.get_price(image_pixels_count)
        )
