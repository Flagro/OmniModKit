from typing import List, Type, Generator, AsyncGenerator

from pydantic import BaseModel

from .ai_config import AIConfig
from .models_tools.base_model_toolkit import BaseModelToolkit


class ModelsToolkit:
    def __init__(
        self,
        openai_api_key: str,
        ai_config: AIConfig,
        models: List[Type[BaseModelToolkit]] = None,
    ):
        self.ai_config = ai_config
        self.models = {model.model_name: model(openai_api_key) for model in models}

    def get_model(self, model_name: str) -> BaseModelToolkit:
        return self.models[model_name]

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
    ) -> Generator[BaseModel]:
        """
        Streams the model with the given input parameters
        """
        yield from self.get_model(model_name).stream(*args, **kwargs)

    async def astream_model(
        self,
        model_name: str,
        *args,
        **kwargs,
    ) -> AsyncGenerator[BaseModel]:
        """
        Streams the model with the given input parameters asynchronously
        """
        async for model in self.get_model(model_name).astream(*args, **kwargs):
            yield model

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
            map(lambda model: model.get_price(*args, **kwargs), self.models.values())
        )
