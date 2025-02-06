from typing import List, Type, Generator, AsyncGenerator

from pydantic import BaseModel

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
        self.tools = {model.model_name: model(openai_api_key) for model in self.models}

    def get_model(self, model_name: str) -> BaseModelToolkit:
        return self.tools[model_name]

    def run_model(
        self,
        model_name: str,
        *args,
        **kwargs,
    ) -> BaseModel:
        """
        Runs the model with the given input parameters
        """
        return self.get_model(model_name).run(*args, **kwargs)

    async def arun_model(
        self,
        model_name: str,
        *args,
        **kwargs,
    ) -> BaseModel:
        """
        Runs the model with the given input parameters asynchronously
        """
        return await self.get_model(model_name).arun(*args, **kwargs)

    def stream_model(
        self,
        model_name: str,
        *args,
        **kwargs,
    ) -> BaseModel:
        """
        Streams the model with the given input parameters
        """
        return self.get_model(model_name).stream(*args, **kwargs)

    async def astream_model(
        self,
        model_name: str,
        *args,
        **kwargs,
    ) -> BaseModel:
        """
        Streams the model with the given input parameters asynchronously
        """
        return await self.get_model(model_name).astream(*args, **kwargs)

    def get_price(
        self,
        *args,
        **kwargs,
    ) -> float:
        """
        Returns the price of the AI services for the given
        input parameters
        """
        return sum(
            map(lambda model: model.get_price(*args, **kwargs), self.tools.values())
        )
