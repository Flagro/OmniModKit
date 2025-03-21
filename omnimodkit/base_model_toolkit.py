import io
import base64
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type, Generator, AsyncGenerator, List
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .prompt_manager import PromptManager
from .ai_config import AIConfig, Model
from .moderation import Moderation


class BaseModelToolkit(ABC):
    model_name: str
    openai_api_key: str

    def __init__(
        self,
        model,
        ai_config: AIConfig,
        prompt_manager: PromptManager,
        openai_api_key: str,
    ):
        self.client = model
        self.ai_config = ai_config
        self.prompt_manager = prompt_manager
        self.openai_api_key = openai_api_key
        self.moderation = Moderation(openai_api_key)

    def get_model_chain(self) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self.openai_api_key,
            temperature=self.get_model().temperature,
            model=self.get_model().name,
            max_tokens=self.get_model().structured_output_max_tokens,
        )

    @staticmethod
    def get_b64_from_bytes(in_memory_stream: io.BytesIO) -> str:
        return base64.b64encode(in_memory_stream.getvalue()).decode()

    def run(
        self,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        *args,
        **kwargs,
    ) -> BaseModel:
        system_prompt = system_prompt or self.get_default_system_prompt()
        pydantic_model = pydantic_model or self.get_default_pydantic_model()
        return self.run_impl(
            system_prompt=system_prompt, pydantic_model=pydantic_model, *args, **kwargs
        )

    @abstractmethod
    def run_impl(
        self, system_prompt: str, pydantic_model: str, *args, **kwargs
    ) -> BaseModel:
        raise NotImplementedError

    async def arun(
        self,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        *args,
        **kwargs,
    ) -> BaseModel:
        system_prompt = system_prompt or self.get_default_system_prompt()
        pydantic_model = pydantic_model or self.get_default_pydantic_model()
        return await self.arun_impl(
            system_prompt=system_prompt, pydantic_model=pydantic_model, *args, **kwargs
        )

    async def arun_impl(
        self, system_prompt: str, pydantic_model: str, *args, **kwargs
    ) -> BaseModel:
        raise NotImplementedError

    def stream(
        self,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        *args,
        **kwargs,
    ) -> Generator[BaseModel]:
        system_prompt = system_prompt or self.get_default_system_prompt()
        pydantic_model = pydantic_model or self.get_default_pydantic_model()
        yield from self.stream_impl(
            system_prompt=system_prompt, pydantic_model=pydantic_model, *args, **kwargs
        )

    def stream_impl(
        self, system_prompt: str, pydantic_model: str, *args, **kwargs
    ) -> Generator[BaseModel]:
        raise NotImplementedError

    async def astream(
        self,
        system_prompt: Optional[str] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[BaseModel]:
        system_prompt = system_prompt or self.get_default_system_prompt()
        pydantic_model = pydantic_model or self.get_default_pydantic_model()
        async for model in self.astream_impl(
            system_prompt=system_prompt, pydantic_model=pydantic_model, *args, **kwargs
        ):
            yield model

    async def astream_impl(
        self, system_prompt: str, pydantic_model: str, *args, **kwargs
    ) -> AsyncGenerator[BaseModel]:
        raise NotImplementedError

    @abstractmethod
    def get_price(*args, **kwargs):
        raise NotImplementedError

    def _get_default_model(self) -> Optional[Model]:
        return next(
            iter(
                filter(
                    lambda model: getattr(model, "default", False),
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

    @staticmethod
    def compose_messages_for_structured_output(
        system_prompt: str, format_instructions: str, input_dict: Dict[str, Any]
    ) -> List[HumanMessage]:
        return [
            HumanMessage(
                content=[
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": format_instructions},
                    input_dict,
                ]
            )
        ]

    def _get_structured_output(
        self,
        input_dict: Dict[str, Any],
        system_prompt: str,
        pydantic_model: Type[BaseModel],
    ) -> BaseModel:
        parser = JsonOutputParser(pydantic_model=pydantic_model)
        model = self.get_model_chain()
        messages = self.compose_messages_for_structured_output(
            system_prompt, parser.get_format_instructions(), input_dict
        )
        msg = model.invoke(messages)
        parsed_output = parser.invoke(msg.content)
        return pydantic_model(**parsed_output)

    async def _aget_structured_output(
        self,
        input_dict: Dict[str, Any],
        system_prompt: str,
        pydantic_model: Type[BaseModel],
    ) -> BaseModel:
        parser = JsonOutputParser(pydantic_model=pydantic_model)
        model = self.get_model_chain()
        messages = self.compose_messages_for_structured_output(
            system_prompt, parser.get_format_instructions(), input_dict
        )
        msg = await model.ainvoke(messages)
        parsed_output = await parser.ainvoke(msg.content)
        return pydantic_model(**parsed_output)

    def get_default_system_prompt(self) -> str:
        return self.prompt_manager.get_default_system_prompt(self.model_name)

    def get_default_pydantic_model(self, *args, **kwargs) -> Type[BaseModel]:
        return self.prompt_manager.get_default_pydantic_model(
            self.model_name, *args, **kwargs
        )
