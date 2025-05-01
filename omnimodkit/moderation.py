from openai import OpenAI, AsyncOpenAI
from .ai_config import AIConfig
from .base_toolkit_model import BaseToolkitModel


class ModerationError(Exception):
    pass


class Moderation(BaseToolkitModel):
    model_name = "moderation"

    def __init__(self, ai_config: AIConfig, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.async_client = AsyncOpenAI(api_key=openai_api_key)
        self.ai_config = ai_config
        self.openai_api_key = openai_api_key
        self.moderation_model = self.get_model().name

    def moderate_text(self, text: str) -> bool:
        """
        Moderates the text and returns True if the text is safe
        """
        input_formatted = text
        response = self.client.moderations.create(
            model=self.moderation_model,
            input=input_formatted,
        )
        flagged = response.results[0].flagged
        return not flagged

    async def amoderate_text(self, text: str) -> bool:
        """
        Moderates the text and returns True if the text is safe
        """
        input_formatted = text
        response = await self.async_client.moderations.create(
            model=self.moderation_model,
            input=input_formatted,
        )
        flagged = response.results[0].flagged
        return not flagged
