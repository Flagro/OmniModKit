from typing import Literal, AsyncGenerator, List, Dict, Generator
import tiktoken
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from ..base_model_toolkit import BaseModelToolkit


class YesOrNoInvalidResponse(Exception):
    pass


class TextModel(BaseModelToolkit):
    def __init__(self, model):
        self.client = model

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
            model=self._get_default_model("text").name,
            messages=messages,
            stream=False,
            temperature=self.get_default_temperature(),
        )
        text_response = response.choices[0].message.content
        return text_response

    async def arun(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.chat.completions.create(
            model=self._get_default_model("text").name,
            messages=messages,
            stream=False,
            temperature=self.get_default_temperature(),
        )
        text_response = response.choices[0].message.content
        return text_response

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str]:
        response = self.llm.chat.completions.create(
            model=self._get_default_model("text").name,
            messages=messages,
            stream=True,
            temperature=self.get_default_temperature(),
        )
        for message in response:
            yield message.choices[0].message.content

    async def astream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str]:
        response = self.llm.chat.completions.create(
            model=self._get_default_model("text").name,
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
        text_response = await self.get_response(self.compose_message_openai(question))
        lower_response = text_response.lower()
        if "yes" in lower_response:
            return True
        if "no" in lower_response:
            return False
        raise YesOrNoInvalidResponse(f"Response: {text_response}")

    @staticmethod
    def count_tokens(text: str) -> int:
        return tiktoken.count(text)
