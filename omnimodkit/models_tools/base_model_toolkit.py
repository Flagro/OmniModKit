from abc import ABC, abstractmethod


class BaseModelToolkit(ABC):
    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError
