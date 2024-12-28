from abc import ABC, abstractmethod
from prompt_manager import PromptManager
from ai_config import AIConfig


class BaseModelToolkit(ABC):
    model_name: str

    def __init__(self, model, ai_config: AIConfig, prompt_manager: PromptManager):
        self.client = model
        self.ai_config = ai_config
        self.prompt_manager = prompt_manager

    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError

    def get_model(self):
        return self._get_default_model(self.model_name)
