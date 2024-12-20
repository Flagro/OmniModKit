from abc import ABC, abstractmethod

from ..models_toolkit import ModelsToolkit


class BaseTool(ABC):
    name: str
    description: str

    def __init__(
        self,
        models_toolkit: ModelsToolkit,
    ):
        self.models_toolkit = models_toolkit

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    async def arun(self, *args, **kwargs):
        raise NotImplementedError
