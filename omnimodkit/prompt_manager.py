from typing import Type, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class DefaultAudioInformation(BaseModel):
    audio_description: str = Field(description="a short description of the audio")

    def __str__(self):
        return self.audio_description


class DefaultImageInformation(BaseModel):
    image_description: str = Field(description="a short description of the image")
    image_type: Literal["screenshot", "picture", "selfie", "anime"] = Field(
        description="type of the image"
    )
    main_objects: List[str] = Field(
        description="list of the main objects on the picture"
    )

    def __str__(self):
        main_objects_prompt = ", ".join(self.main_objects)
        return (
            f'Image description: "{self.image_description}", '
            f'Image type: "{self.image_type}", '
            f'Main objects: "{main_objects_prompt}"'
        )


class DefaultImage(BaseModel):
    image_url: str = Field(description="url of the image")


class DefaultText(BaseModel):
    text: str = Field(description="text to be processed")


class DefaultTextChunk(BaseModel):
    text_chunk: str = Field(description="text chunk to be processed")


class PromptManager:
    @staticmethod
    def get_default_audio_information() -> Type[BaseModel]:
        return DefaultAudioInformation

    @staticmethod
    def get_default_image_information() -> Type[BaseModel]:
        return DefaultImageInformation

    @staticmethod
    def get_current_date_prompt() -> str:
        date_prompt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return date_prompt
