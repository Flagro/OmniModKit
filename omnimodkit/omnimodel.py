import os
from typing import Optional
from .ai_config import AIConfig
from .audio_recognition_model.audio_recognition_model import (
    AudioRecognitionModel,
)
from .image_generation_model.image_generation_model import ImageGenerationModel
from .text_model.text_model import TextModel
from .vision_model.vision_model import VisionModel
from .moderation import Moderation


class OmniModel:
    def __init__(
        self, openai_api_key: Optional[str] = None, ai_config: Optional[AIConfig] = None
    ):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set in the environment! "
                    "Set it for these integration tests."
                )
        if ai_config is None:
            try:
                ai_config = AIConfig.load("ai_config.yaml")
            except FileNotFoundError:
                raise ValueError(
                    "ai_config.yaml file not found! "
                    "Set it for these integration tests."
                )
        self.openai_api_key = openai_api_key
        self.ai_config = ai_config
        self.text_model = TextModel(ai_config=ai_config, openai_api_key=openai_api_key)
        self.vision_model = VisionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.image_generation_model = ImageGenerationModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.audio_recognition_model = AudioRecognitionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.moderation_model = Moderation(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
