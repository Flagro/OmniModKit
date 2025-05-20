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


class ModelsToolkit:
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
        self._text_model: Optional[TextModel] = None
        self._vision_model: Optional[VisionModel] = None
        self._image_generation_model: Optional[ImageGenerationModel] = None
        self._audio_recognition_model: Optional[AudioRecognitionModel] = None
        self._moderation_model: Optional[Moderation] = None

    @property
    def text_model(self) -> TextModel:
        if self._text_model is None:
            self._text_model = TextModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._text_model

    @property
    def vision_model(self) -> VisionModel:
        if self._vision_model is None:
            self._vision_model = VisionModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._vision_model

    @property
    def image_generation_model(self) -> ImageGenerationModel:
        if self._image_generation_model is None:
            self._image_generation_model = ImageGenerationModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._image_generation_model

    @property
    def audio_recognition_model(self) -> AudioRecognitionModel:
        if self._audio_recognition_model is None:
            self._audio_recognition_model = AudioRecognitionModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._audio_recognition_model

    @property
    def moderation_model(self):
        if self._moderation_model is None:
            self._moderation_model = Moderation(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._moderation_model

    def get_price(*args, **kwargs):
        """
        Get the price of the model
        """
        return sum(
            model.get_price(*args, **kwargs)
            for model in [
                TextModel,
                VisionModel,
                ImageGenerationModel,
                AudioRecognitionModel,
            ]
        )
