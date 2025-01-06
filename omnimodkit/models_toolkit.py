from typing import List, Type

from .ai_config import AIConfig
from .models_tools.base_model_toolkit import BaseModelToolkit
from .models_tools import (
    TextModel,
    VisionModel,
    ImageGenerationModel,
    AudioRecognitionModel,
)


class ModelsToolkit:
    models: List[Type[BaseModelToolkit]] = [
        TextModel,
        VisionModel,
        ImageGenerationModel,
        AudioRecognitionModel,
    ]

    def __init__(self, openai_api_key: str, ai_config: AIConfig):
        self.ai_config = ai_config
        self.tools = [model(openai_api_key) for model in self.models]

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
