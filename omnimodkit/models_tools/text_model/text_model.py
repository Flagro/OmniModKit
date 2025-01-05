from typing import Literal, AsyncGenerator, List, Dict, Generator
import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from ..base_model_toolkit import BaseModelToolkit
from ...ai_config import Model


class YesOrNoInvalidResponse(Exception):
    pass


class TextModel(BaseModelToolkit):
    model_name = "text"

    def __init__(self, openai_api_key):
        self.llm = OpenAI(api_key=openai_api_key, model=self.get_model().name)

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.TextGeneration.Models

    @staticmethod
    def compose_message_openai(
        message_text: str, role: Literal["user", "system"] = "user"
    ) -> List[Dict[str, str]]:
        return [{"role": role, "content": message_text}]

    def get_default_temperature(self) -> float:
        return self.ai_config.TextGeneration.temperature

    @staticmethod
    def compose_messages_openai(
        user_input: str, system_prompt: str
    ) -> List[Dict[str, str]]:
        return TextModel.compose_message_openai(
            system_prompt, role="system"
        ) + TextModel.compose_message_openai(user_input, role="user")

    def run(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=False,
            temperature=self.get_default_temperature(),
        )
        text_response = response.choices[0].message.content
        return text_response

    async def arun(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=False,
            temperature=self.get_default_temperature(),
        )
        text_response = response.choices[0].message.content
        return text_response

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str]:
        response = self.llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=True,
            temperature=self.get_default_temperature(),
        )
        for message in response:
            yield message.choices[0].message.content

    async def astream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str]:
        response = self.llm.chat.completions.create(
            model=self.get_model().name,
            messages=messages,
            stream=True,
            temperature=self.get_default_temperature(),
        )
        async for message in response:
            yield message.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(YesOrNoInvalidResponse),
    )
    async def ask_yes_no_question(self, question: str) -> bool:
        text_response = await self.arun(self.compose_message_openai(question))
        lower_response = text_response.lower()
        if "yes" in lower_response:
            return True
        if "no" in lower_response:
            return False
        raise YesOrNoInvalidResponse(f"Response: {text_response}")

    def count_tokens(self, text: str) -> int:
        encoding_for_model = tiktoken.encoding_for_model(self.get_model().name)
        nums_tokens = len(encoding_for_model.encode(text))
        return nums_tokens

    def get_price(
        self,
        token_len: int,
        *args,
        **kwargs,
    ) -> float:
        input_token_price = self.get_model().rate.input_token_price
        output_token_price = self.get_model().rate.output_token_price

        return token_len * input_token_price + token_len * output_token_price
