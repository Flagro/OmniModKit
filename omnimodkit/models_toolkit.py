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
        *args,
        **kwargs,
    ) -> float:
        """
        Returns the price of the AI services for the given
        input parameters
        """
        return (
            self.audio_recognition_model.get_price(*args, **kwargs)
            + self.image_generation_model.get_price(*args, **kwargs)
            + self.text_model.get_price(*args, **kwargs)
            + self.vision_model.get_price(*args, **kwargs)
        )
