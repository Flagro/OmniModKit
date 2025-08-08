import os
import io
from typing import Optional, Literal, List
from .ai_config import AIConfig
from .audio_recognition_model.audio_recognition_model import (
    AudioRecognitionModel,
)
from .audio_generation_model.audio_generation_model import AudioGenerationModel
from .image_generation_model.image_generation_model import ImageGenerationModel
from .text_model.text_model import TextModel
from .vision_model.vision_model import VisionModel
from .moderation import Moderation


AvailableModelType = Literal[
    "text", "vision", "image_generation", "audio_recognition", "audio_generation"
]


class ModelsToolkit:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        ai_config: Optional[AIConfig] = None,
        allowed_models: Optional[List[AvailableModelType]] = None,
        allow_default_ai_config: bool = False,
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
                if allow_default_ai_config:
                    ai_config = AIConfig.load_default()
                else:
                    raise ValueError(
                        "ai_config.yaml file not found! "
                        "To use a default configuration, set allow_default_ai_config=True."
                    )
        self.openai_api_key = openai_api_key
        self.ai_config = ai_config
        self.allowed_models = allowed_models
        self._text_model: Optional[TextModel] = None
        self._vision_model: Optional[VisionModel] = None
        self._image_generation_model: Optional[ImageGenerationModel] = None
        self._audio_recognition_model: Optional[AudioRecognitionModel] = None
        self._audio_generation_model: Optional[AudioGenerationModel] = None
        self._moderation_model: Optional[Moderation] = None

    @property
    def text_model(self) -> TextModel:
        if not self.can_use_model("text"):
            raise ValueError("Text model is not allowed.")
        if self._text_model is None:
            self._text_model = TextModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._text_model

    @property
    def vision_model(self) -> VisionModel:
        if not self.can_use_model("vision"):
            raise ValueError("Vision model is not allowed.")
        if self._vision_model is None:
            self._vision_model = VisionModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._vision_model

    @property
    def image_generation_model(self) -> ImageGenerationModel:
        if not self.can_use_model("image_generation"):
            raise ValueError("Image generation model is not allowed.")
        if self._image_generation_model is None:
            self._image_generation_model = ImageGenerationModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._image_generation_model

    @property
    def audio_recognition_model(self) -> AudioRecognitionModel:
        if not self.can_use_model("audio_recognition"):
            raise ValueError("Audio recognition model is not allowed.")
        if self._audio_recognition_model is None:
            self._audio_recognition_model = AudioRecognitionModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._audio_recognition_model

    @property
    def audio_generation_model(self) -> AudioGenerationModel:
        if not self.can_use_model("audio_generation"):
            raise ValueError("Audio generation model is not allowed.")
        if self._audio_generation_model is None:
            self._audio_generation_model = AudioGenerationModel(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._audio_generation_model

    @property
    def moderation_model(self):
        if self._moderation_model is None:
            self._moderation_model = Moderation(
                ai_config=self.ai_config, openai_api_key=self.openai_api_key
            )
        return self._moderation_model

    def all_models_allowed(self) -> bool:
        """
        Check if all model types are allowed.
        """
        return self.allowed_models is None

    def can_use_model(self, model_type: AvailableModelType) -> bool:
        """
        Check if a specific model type is allowed.
        """
        return self.all_models_allowed() or model_type in self.allowed_models

    def get_price(
        self,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        input_image: Optional[io.BytesIO] = None,
        output_image_url: Optional[str] = None,
        input_audio: Optional[io.BytesIO] = None,
        output_audio: Optional[io.BytesIO] = None,
    ):
        """
        Get the price of the model
        """
        total_price = 0.0
        if input_text is not None or output_text is not None:
            total_price += self.text_model.get_price(
                input_text=input_text, output_text=output_text
            )
        if output_image_url is not None:
            total_price += self.image_generation_model.get_price(
                input_text=input_text, output_image_url=output_image_url
            )
        if input_image is not None:
            total_price += self.vision_model.get_price(
                input_image=input_image, output_text=output_text
            )
        if output_audio is not None:
            total_price += self.audio_generation_model.get_price(
                input_text=input_text, output_audio=output_audio
            )
        if input_audio is not None:
            total_price += self.audio_recognition_model.get_price(
                input_audio=input_audio, output_text=output_text
            )
        return total_price

    def estimate_price(
        self,
        input_text: Optional[str] = None,
        input_image: Optional[io.BytesIO] = None,
        input_audio: Optional[io.BytesIO] = None,
        enable_text_output: bool = True,
        enable_image_generation: bool = False,
        enable_audio_generation: bool = False,
    ):
        """
        Estimate the price of the model based on inputs and expected output types.
        Uses dummy output values for estimation.
        Only includes pricing for models that are allowed based on the allowed_models configuration.
        """
        total_price = 0.0

        # Text model pricing (input + estimated output)
        if (input_text is not None or enable_text_output) and self.can_use_model("text"):
            estimated_output_text = input_text if enable_text_output else None
            total_price += self.text_model.get_price(
                input_text=input_text, output_text=estimated_output_text
            )

        # Image generation pricing
        if enable_image_generation and self.can_use_model("image_generation"):
            total_price += self.image_generation_model.get_price(
                input_text=input_text, output_image_url="dummy_image_url"
            )

        # Vision model pricing (input image)
        if input_image is not None and self.can_use_model("vision"):
            estimated_output_text = "dummy_output_text" if enable_text_output else None
            total_price += self.vision_model.get_price(
                input_image=input_image, output_text=estimated_output_text
            )

        # Audio generation pricing
        if enable_audio_generation and self.can_use_model("audio_generation"):
            total_price += self.audio_generation_model.get_price(
                input_text=input_text, output_audio=io.BytesIO(b"dummy_audio")
            )

        # Audio recognition pricing (input audio)
        if input_audio is not None and self.can_use_model("audio_recognition"):
            estimated_output_text = "dummy_output_text" if enable_text_output else None
            total_price += self.audio_recognition_model.get_price(
                input_audio=input_audio, output_text=estimated_output_text
            )

        return total_price
