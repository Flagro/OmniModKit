from .ai_config import AIConfig
from .models_tools import (
    TextModel,
    VisionModel,
    ImageGenerationModel,
    AudioRecognitionModel,
)
from .models_toolkit import ModelsToolkit


class UniversalModelsToolkit:
    def __init__(self, openai_api_key: str, ai_config: AIConfig):
        self.models_toolkit = ModelsToolkit(
            openai_api_key,
            ai_config,
            [
                TextModel,
                VisionModel,
                ImageGenerationModel,
                AudioRecognitionModel,
            ],
        )
