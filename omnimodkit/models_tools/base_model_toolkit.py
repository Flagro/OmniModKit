from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type
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

    @property
    def default_attribute(self) -> str:
        return f"{self.model_name}_default"

    @abstractmethod
    def run(*args, **kwargs):
        raise NotImplementedError

    async def arun(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError

    def _get_default_model(self) -> Optional[Model]:
        return next(
            iter(
                filter(
                    lambda model: getattr(model, self.default_attribute, False),
                    self.get_models_dict().values(),
                )
            ),
            None,
        )

    @abstractmethod
    def get_models_dict(self) -> Dict[str, Model]:
        raise NotImplementedError

    def get_model(self, model_name: Optional[str] = None) -> Model:
        if model_name is None:
            return self._get_default_model()
        model = self.get_models_dict().get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        return model

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

    async def aget_structured_output(
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
        msg = await model.ainvoke(
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
        parsed_output = parser.ainvoke(contents)
        return pydantic_object(**parsed_output)
