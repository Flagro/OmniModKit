from typing import (
    AsyncGenerator,
    List,
    Generator,
    Optional,
    Type,
)
from pydantic import BaseModel, Field

from ..base_toolkit_model import BaseToolkitModel, OpenAIMessage
from ..ai_config import TextGeneration
from ..moderation import ModerationError


class YesNoResponse(BaseModel):
    answer_is_yes: bool


class DefaultText(BaseModel):
    text: str = Field(description="text to be processed")

    def __str__(self):
        return self.text


class DefaultTextChunk(BaseModel):
    text_chunk: str = Field(description="text chunk to be processed")

    def __str__(self):
        return self.text_chunk


class TextModel(BaseToolkitModel[DefaultText, DefaultTextChunk]):
    model_name = "text"
    default_pydantic_model: Type[DefaultText] = DefaultText
    default_streamable_pydantic_model: Optional[Type[DefaultTextChunk]] = (
        DefaultTextChunk
    )

    def get_default_system_prompt(self) -> str:
        return "Please provide the necessary information."

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
        pydantic_model = self.default_streamable_pydantic_model
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
        pydantic_model = self.default_streamable_pydantic_model
        messages = self._compose_messages_list(
            user_input, system_prompt, communication_history
        )
        await self.amoderate_messages(messages)
        llm = self.get_langchain_llm()
        langchain_messages = self.get_langchain_messages(messages)
        async for message in llm.astream(langchain_messages):
            yield pydantic_model(text_chunk=message.content)

    def ask_yes_no_question(self, question: str) -> bool:
        response: YesNoResponse = self.run(
            self.compose_message_openai(question), YesNoResponse
        )
        return response.answer_is_yes

    async def async_ask_yes_no_question(self, question: str) -> bool:
        response: YesNoResponse = await self.arun(
            self.compose_message_openai(question), YesNoResponse
        )
        return response.answer_is_yes

    def get_price(
        self,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
    ) -> float:
        price = 0.0
        if input_text is not None:
            input_token_len = self.count_tokens(input_text)
            input_token_price = self.get_model().rate.input_token_price
            price += input_token_len * input_token_price
        if output_text is not None:
            output_token_len = self.count_tokens(output_text)
            output_token_price = self.get_model().rate.output_token_price
            price += output_token_len * output_token_price
        return price
