import io
from typing import AsyncIterator
from openai import OpenAI

from .ai_config import AIConfig
from .moderation import ModerationError
from .prompt_manager import PromptManager
from .ai_utils.describe_image import DescribeImageUtililty
from .ai_utils.describe_audio import DescribeAudioUtililty
from .models_toolkit import ModelsToolkit
from .moderation import Moderation
from .agent_tools.agent_toolkit import AIAgentToolkit


class AI:
    def __init__(
        self, openai_api_key: str, ai_config: AIConfig, prompt_manager: PromptManager
    ):
        moderation_model = OpenAI(api_key=openai_api_key)
        self.moderation = Moderation(model=moderation_model)
        self.ai_config = ai_config
        self.prompt_manager = prompt_manager
        self.models_toolkit = ModelsToolkit(openai_api_key, ai_config)

    async def describe_image(
        self,
        in_memory_image_stream: io.BytesIO,
    ) -> str:
        # TODO: person, context, message likely not needed for utility functions
        if not self.moderation.moderate_image(in_memory_image_stream):
            raise ModerationError("Image is not safe")
        utility = DescribeImageUtililty(self.models_toolkit)
        image_information = await utility.arun(in_memory_image_stream)
        image_description = await self.prompt_manager.compose_image_description_prompt(
            image_information
        )
        return image_description

    async def transcribe_audio(
        self,
        in_memory_audio_stream: io.BytesIO,
    ) -> str:
        if not self.moderation.moderate_audio(in_memory_audio_stream):
            raise ModerationError("Audio is not safe")
        utility = DescribeAudioUtililty(self.models_toolkit)
        audio_information = await utility.arun(in_memory_audio_stream)
        audio_description = await self.prompt_manager.compose_audio_description_prompt(
            audio_information
        )
        return audio_description

    async def generate_image(
        self, prompt: str
    ) -> str:
        """
        Returns the URL of the generated image
        """
        toolkit = AIAgentToolkit(
            self.models_toolkit, self.prompt_manager
        )
        return await toolkit.image_generator.arun(prompt)

    async def get_reply(self, user_input: str, system_prompt: str) -> str:
        if not self.moderation.moderate_text(user_input):
            raise ModerationError("Text is not safe")
        messages = self.models_toolkit.compose_messages_openai(
            user_input, system_prompt
        )
        return await self.models_toolkit.get_response(messages)

    async def get_streaming_reply(
        self, user_input: str, system_prompt: str
    ) -> AsyncIterator[str]:
        if not self.moderation.moderate_text(user_input):
            raise ModerationError("Text is not safe")
        messages = self.models_toolkit.compose_messages_openai(
            user_input, system_prompt
        )
        async for response in self.models_toolkit.get_streaming_response(messages):
            yield response
