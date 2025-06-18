import io
from typing import Type, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class DefaultAudioInformation(BaseModel):
    audio_description: str = Field(description="a short description of the audio")

    def __str__(self):
        return self.audio_description


class DefaultAudio(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    audio_bytes: io.BytesIO = Field(description="in-memory audio bytes in ogg format")

    def __str__(self):
        return f"Audio bytes: {self.audio_bytes.name} ({self.audio_bytes.getbuffer().nbytes} bytes)"


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

    def __str__(self):
        return f"Image url: {self.image_url}"


class DefaultText(BaseModel):
    text: str = Field(description="text to be processed")

    def __str__(self):
        return self.text


class DefaultTextChunk(BaseModel):
    text_chunk: str = Field(description="text chunk to be processed")

    def __str__(self):
        return self.text_chunk


class PromptManager:
    @staticmethod
    def get_default_audio_information() -> Type[BaseModel]:
        return DefaultAudioInformation

    @staticmethod
    def get_default_audio() -> Type[BaseModel]:
        return DefaultAudio

    @staticmethod
    def get_default_image_information() -> Type[BaseModel]:
        return DefaultImageInformation

    @staticmethod
    def get_default_image() -> Type[BaseModel]:
        return DefaultImage

    @staticmethod
    def get_default_text() -> Type[BaseModel]:
        return DefaultText

    @staticmethod
    def get_default_text_chunk() -> Type[BaseModel]:
        return DefaultTextChunk

    @staticmethod
    def get_current_date_prompt() -> str:
        date_prompt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return date_prompt

    @staticmethod
    def get_default_pydantic_model(
        model_name: str, streamable: bool = False
    ) -> Type[BaseModel]:
        if model_name == "audio_recognition":
            return PromptManager.get_default_audio_information()
        if model_name == "audio_generation":
            return PromptManager.get_default_audio()
        if model_name == "vision":
            return PromptManager.get_default_image_information()
        if model_name == "image_generation":
            return PromptManager.get_default_image()
        if streamable:
            return PromptManager.get_default_text_chunk()
        return PromptManager.get_default_text()
