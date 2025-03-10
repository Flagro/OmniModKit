import functools
from typing import Literal, AsyncGenerator, List, Dict, Generator, Optional, TypedDict
import tiktoken
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from openai import OpenAI

from ..base_model_toolkit import BaseModelToolkit
from ..ai_config import Model
from ..moderation import ModerationError


@functools.lru_cache()
def _get_encoding_for_model(model_name: str) -> tiktoken.Encoding:
    """
    Returns the tiktoken encoding for the given model name,
    leveraging LRU caching to avoid repeated calls.
    """
    return tiktoken.encoding_for_model(model_name)


class YesNoResponse(BaseModel):
    answer_is_yes: bool


class OpenAIMessage(TypedDict):
    role: str
    content: str


class TextModel(BaseModelToolkit):
    model_name = "text"

    def get_langchain_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self.openai_api_key,
            temperature=self.get_default_temperature(),
            model=self.get_model().name,
        )

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.TextGeneration.Models

    @staticmethod
    def compose_message_openai(
        message_text: str, role: Literal["user", "system"] = "user"
    ) -> OpenAIMessage:
        return OpenAIMessage({"role": role, "content": message_text})

    def get_default_temperature(self) -> float:
        return self.get_model().temperature

    def compose_messages_openai(
        self, user_input: str, system_prompt: Optional[str] = None
    ) -> List[OpenAIMessage]:
        # TODO: if system prompt is not provided, just don't include it
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()
        return [
            TextModel.compose_message_openai(system_prompt, role="system"),
            TextModel.compose_message_openai(user_input),
        ]

    @staticmethod
    def get_langchain_message(message_dict: OpenAIMessage) -> BaseMessage:
        return (
            HumanMessage(content=message_dict["content"])
            if message_dict["role"] == "user"
            else SystemMessage(content=message_dict["content"])
        )

    @staticmethod
    def check_system_prompt_in_messages(messages: List[OpenAIMessage]) -> bool:
        return any(message["role"] == "system" for message in messages)

    @staticmethod
    def get_langchain_messages(messages: List[OpenAIMessage]) -> List[BaseMessage]:
        return list(map(TextModel.get_langchain_message, messages))

    def run(
        self, messages: List[OpenAIMessage], pydantic_model: Optional[BaseModel] = None
    ) -> BaseModel:
        # TODO: run moderation for each message
        if (
            self.ai_config.TextGeneration.moderation_needed
            and not self.moderation.moderate_text(messages[-1]["content"])
        ):
            raise ModerationError(
                f"Text description '{messages[-1]['content']}' was rejected by the moderation system"
            )
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        llm = self.get_langchain_llm()
        structured_llm = llm.with_structured_output(pydantic_model)
        langchain_messages = TextModel.get_langchain_messages(messages)
        structured_response = structured_llm.invoke(langchain_messages)
        return structured_response

    async def arun(
        self, messages: List[OpenAIMessage], pydantic_model: Optional[BaseModel] = None
    ) -> BaseModel:
        # TODO: run moderation for each message
        if (
            self.ai_config.TextGeneration.moderation_needed
            and not self.moderation.moderate_text(messages[-1]["content"])
        ):
            raise ModerationError(
                f"Text description '{messages[-1]['content']}' was rejected by the moderation system"
            )
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        llm = self.get_langchain_llm()
        structured_llm = llm.with_structured_output(pydantic_model)
        langchain_messages = TextModel.get_langchain_messages(messages)
        structured_response = await structured_llm.ainvoke(langchain_messages)
        return structured_response

    def stream(
        self, messages: List[OpenAIMessage], pydantic_model: Optional[BaseModel] = None
    ) -> Generator[BaseModel]:
        # TODO: run moderation for each message
        if (
            self.ai_config.TextGeneration.moderation_needed
            and not self.moderation.moderate_text(messages[-1]["content"])
        ):
            raise ModerationError(
                f"Text description '{messages[-1]['content']}' was rejected by the moderation system"
            )
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model(streamable=True)
        else:
            # TODO: fix this
            raise ValueError("pydantic_model is not supported for streaming")
        llm = OpenAI(api_key=self.openai_api_key, model=self.get_model().name)
        response = llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=True,
            temperature=self.get_default_temperature(),
        )
        for message in response:
            yield pydantic_model(message.choices[0].message.content)

    async def astream(
        self, messages: List[OpenAIMessage], pydantic_model: Optional[BaseModel] = None
    ) -> AsyncGenerator[BaseModel]:
        # TODO: run moderation for each message
        if (
            self.ai_config.TextGeneration.moderation_needed
            and not self.moderation.moderate_text(messages[-1]["content"])
        ):
            raise ModerationError(
                f"Text description '{messages[-1]['content']}' was rejected by the moderation system"
            )
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model(streamable=True)
        else:
            # TODO: fix this
            raise ValueError("pydantic_model is not supported for streaming")
        llm = OpenAI(api_key=self.openai_api_key, model=self.get_model().name)
        response = llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=True,
            temperature=self.get_default_temperature(),
        )
        async for message in response:
            yield pydantic_model(message.choices[0].message.content)

    async def ask_yes_no_question(self, question: str) -> bool:
        response: YesNoResponse = await self.arun(
            self.compose_message_openai(question), YesNoResponse
        )
        return response.answer_is_yes

    def count_tokens(self, text: str) -> int:
        encoding = _get_encoding_for_model(self.get_model().name)
        encoded_text = encoding.encode(text)
        return len(encoded_text)

    def get_price(
        self,
        input_token_len: int,
        output_token_len: int,
        *args,
        **kwargs,
    ) -> float:
        input_token_price = self.get_model().rate.input_token_price
        output_token_price = self.get_model().rate.output_token_price
        return (
            input_token_len * input_token_price + output_token_len * output_token_price
        )
