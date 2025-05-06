import functools
from typing import (
    AsyncGenerator,
    List,
    Generator,
    Optional,
    Type,
)
import tiktoken
from pydantic import BaseModel

from ..base_toolkit_model import BaseToolkitModel, OpenAIMessage
from ..ai_config import TextGeneration
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


class TextModel(BaseToolkitModel):
    model_name = "text"

    def get_model_config(self) -> TextGeneration:
        return self.ai_config.text_generation

    def moderate_messages(
        self,
        messages: List[OpenAIMessage],
        raise_error: bool = True,
        moderate_system_messages: bool = False,
    ) -> bool:
        if self.moderation_needed:
            for message in messages:
                # System messages are not moderated if moderate_system_messages is False
                if not moderate_system_messages and message["role"] == "system":
                    continue
                if not self.moderate_text(message["content"]):
                    if raise_error:
                        raise ModerationError(
                            f"Text description '{message['content']}' was rejected by the moderation system"
                        )
                    return False
        return True

    async def amoderate_messages(
        self,
        messages: List[OpenAIMessage],
        raise_error: bool = True,
        moderate_system_messages: bool = False,
    ) -> bool:
        if self.moderation_needed:
            for message in messages:
                # System messages are not moderated if moderate_system_messages is False
                if not moderate_system_messages and message["role"] == "system":
                    continue
                if not await self.amoderate_text(message["content"]):
                    if raise_error:
                        raise ModerationError(
                            f"Text description '{message['content']}' was rejected by the moderation system"
                        )
                    return False
        return True

    def run_impl(
        self,
        user_input: str,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: Optional[List[OpenAIMessage]] = None,
    ) -> BaseModel:
        if communication_history is None:
            communication_history = []
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        messages = self._compose_messages_list(
            user_input, system_prompt, communication_history
        )
        self.moderate_messages(messages)
        llm = self.get_langchain_llm()
        structured_llm = llm.with_structured_output(pydantic_model)
        langchain_messages = self.get_langchain_messages(messages)
        structured_response = structured_llm.invoke(langchain_messages)
        return structured_response

    async def arun_impl(
        self,
        user_input: str,
        system_prompt: str,
        pydantic_model: Type[BaseModel],
        communication_history: Optional[List[OpenAIMessage]] = None,
    ) -> BaseModel:
        if communication_history is None:
            communication_history = []
        if pydantic_model is None:
            pydantic_model = self.get_default_pydantic_model()
        messages = self._compose_messages_list(
            user_input, system_prompt, communication_history
        )
        await self.amoderate_messages(messages)
        llm = self.get_langchain_llm()
        structured_llm = llm.with_structured_output(pydantic_model)
        langchain_messages = self.get_langchain_messages(messages)
        structured_response = await structured_llm.ainvoke(langchain_messages)
        return structured_response

    def stream_impl(
        self,
        user_input: str,
        system_prompt: str,
        communication_history: Optional[List[OpenAIMessage]] = None,
    ) -> Generator[BaseModel, None, None]:
        if communication_history is None:
            communication_history = []
        pydantic_model = self.get_default_pydantic_model(streamable=True)
        messages = self._compose_messages_list(
            user_input, system_prompt, communication_history
        )
        self.moderate_messages(messages)
        llm = self.get_langchain_llm()
        langchain_messages = self.get_langchain_messages(messages)
        for message in llm.stream(langchain_messages):
            yield pydantic_model(text_chunk=message.content)

    async def astream_impl(
        self,
        user_input: str,
        system_prompt: str,
        communication_history: Optional[List[OpenAIMessage]] = None,
    ) -> AsyncGenerator[BaseModel, None]:
        if communication_history is None:
            communication_history = []
        pydantic_model = self.get_default_pydantic_model(streamable=True)
        messages = self._compose_messages_list(
            user_input, system_prompt, communication_history
        )
        await self.amoderate_messages(messages)
        llm = self.get_langchain_llm()
        langchain_messages = self.get_langchain_messages(messages)
        async for message in llm.astream(langchain_messages):
            yield pydantic_model(text_chunk=message.content)

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
        input: str,
        output: str,
        *args,
        **kwargs,
    ) -> float:
        input_token_len = self.count_tokens(input)
        output_token_len = self.count_tokens(output)
        input_token_price = self.get_model().rate.input_token_price
        output_token_price = self.get_model().rate.output_token_price
        return (
            input_token_len * input_token_price + output_token_len * output_token_price
        )
