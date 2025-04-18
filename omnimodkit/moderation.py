from typing import Optional
from openai import OpenAI, AsyncOpenAI


class ModerationError(Exception):
    pass


class Moderation:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.async_client = AsyncOpenAI(api_key=openai_api_key)
        self.moderation_model = "omni-moderation-latest"

    def moderate_text(self, text: str, input_description: Optional[str] = None) -> bool:
        """
        Moderates the text and returns True if the text is safe
        """
        input_formatted = f"{input_description}: {text}" if input_description else text
        response = self.client.moderations.create(
            model=self.moderation_model,
            input=input_formatted,
        )
        flagged = response.results[0].flagged
        if flagged:
            return False
        return True

    async def amoderate_text(
        self, text: str, input_description: Optional[str] = None
    ) -> bool:
        """
        Moderates the text and returns True if the text is safe
        """
        input_formatted = f"{input_description}: {text}" if input_description else text
        response = await self.async_client.moderations.create(
            model=self.moderation_model,
            input=input_formatted,
        )
        flagged = response.results[0].flagged
        if flagged:
            return False
        return True
