from abc import ABC, abstractmethod


class BaseModelToolkit(ABC):
    model_name: str

    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError
