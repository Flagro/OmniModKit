from typing import Optional
from pydantic import BaseModel
from .ai_config import AIConfig
from .models_tools import (
    TextModel,
    VisionModel,
    ImageGenerationModel,
    AudioRecognitionModel,
)
from .models_toolkit import ModelsToolkit
from .prompt_manager import PromptManager


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

    def get_text_response(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> BaseModel:
        if system_prompt is None:
            system_prompt = PromptManager.get_default_system_prompt_text()
        messages = TextModel.compose_messages_openai(user_input, system_prompt)
        return self.models_toolkit.run_model("text", messages)
