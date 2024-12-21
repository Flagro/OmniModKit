from abc import ABC, abstractmethod


class BaseUtility(ABC):
    @abstractmethod
    async def run(self, *args, **kwargs):
        raise NotImplementedError

    async def arun(self, *args, **kwargs):
        raise NotImplementedError
