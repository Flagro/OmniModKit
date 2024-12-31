from abc import ABC, abstractmethod
from typing import Optional, Literal, Dict, Any, Type
from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ..prompt_manager import PromptManager
from ..ai_config import AIConfig, Model


class BaseModelToolkit(ABC):
    model_name: str

    def __init__(self, model, ai_config: AIConfig, prompt_manager: PromptManager):
        self.client = model
        self.ai_config = ai_config
        self.prompt_manager = prompt_manager

    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError

    def _get_default_model(self) -> Optional[Model]:
        first_model = None
        for model in self.get_models_dict().values():
            if getattr(model, self.get_default_attr()):
                return model
            if first_model is None:
                first_model = model
        return first_model

    @abstractmethod
    def get_models_dict(self) -> Dict[str, Model]:
        raise NotImplementedError

    @abstractmethod
    def get_default_attr(self) -> str:
        raise NotImplementedError

    def get_model(self) -> Model:
        return self._get_default_model()

    def get_structured_output(
        self,
        input_dict: Dict[str, Any],
        system_prompt: str,
        pydantic_object: Type[BaseModel],
    ) -> Dict[str, Any]:
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        model = ChatOpenAI(
            temperature=self.get_model().temperature,
            model=self.get_model().name,
            max_tokens=1024,
        )
        msg = model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": system_prompt},
                        {"type": "text", "text": parser.get_format_instructions()},
                        input_dict,
                    ]
                )
            ]
        )
        contents = msg.content
        parsed_output = parser.invoke(contents)
        return pydantic_object(**parsed_output)
