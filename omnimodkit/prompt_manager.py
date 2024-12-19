from datetime import datetime

from .ai_utils.describe_image import ImageInformation
from .ai_utils.describe_audio import AudioInformation


class PromptManager:
    async def compose_image_description_prompt(
        self, image_information: ImageInformation
    ) -> str:
        return f"The description of the image is: {image_information}"

    async def compose_audio_description_prompt(
        self, audio_description: AudioInformation
    ) -> str:
        return f"The description of the audio is: {audio_description}"

    @staticmethod
    def get_current_date_prompt() -> str:
        date_prompt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return date_prompt
