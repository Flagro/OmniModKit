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
    def get_default_system_prompt_text() -> str:
        return "Please provide the necessary information."

    @staticmethod
    def get_default_system_prompt_audio() -> str:
        return "Based on the audio, fill out the provided fields."

    @staticmethod
    def get_default_system_prompt_vision() -> str:
        return "Based on the image, fill out the provided fields."

    @staticmethod
    def get_default_system_prompt_image() -> str:
        # TODO: this should not be a formatted string
        return "Please provide the necessary information: {image_desc}"

    @staticmethod
    def get_default_system_prompt(model_name: str) -> str:
        if model_name == "audio_recognition":
            return PromptManager.get_default_system_prompt_audio()
        if model_name == "vision":
            return PromptManager.get_default_system_prompt_vision()
        if model_name == "image_generation":
            return PromptManager.get_default_system_prompt_image()
        return PromptManager.get_default_system_prompt_text()

    @staticmethod
    def get_default_pydantic_model(
        model_name: str, streamable: bool = False
    ) -> Type[BaseModel]:
        if model_name == "audio_recognition":
            return PromptManager.get_default_audio_information()
        if model_name == "vision":
            return PromptManager.get_default_image_information()
        if model_name == "image_generation":
            return PromptManager.get_default_image()
        if streamable:
            return PromptManager.get_default_text_chunk()
        return PromptManager.get_default_text()
