import os
import io
from typing import Optional, Literal
from pydantic import BaseModel, Field
from typing import List, Dict
from .ai_config import AIConfig
from .audio_recognition_model.audio_recognition_model import (
    AudioRecognitionModel,
)
from .image_generation_model.image_generation_model import ImageGenerationModel
from .text_model.text_model import TextModel
from .vision_model.vision_model import VisionModel
from .moderation import Moderation


class OmniModelInput(BaseModel):
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to be used for the model.",
    )
    user_input: Optional[str] = Field(
        default="",
        description="User input to be processed by the model.",
    )
    messages_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of messages history. Each message is a dictionary with 'role' and 'content'.",
    )
    image_description: Optional[str] = Field(
        default=None,
        description="Description of the image to be used for image generation.",
    )
    audio_description: Optional[str] = Field(
        default=None,
        description="Description of the audio to be used for audio recognition.",
    )


class OmniModelOutputType(BaseModel):
    output_type: Literal["text", "image", "audio", "text_with_image"] = Field(
        default="text",
        description="Type of output expected from the model.",
    )


class OmniModelOutput(BaseModel):
    text_response: Optional[str] = Field(
        default=None,
        description="Text response from the model.",
    )
    image_response: Optional[io.BytesIO] = Field(
        default=None,
        description="In-memory image stream for image generation.",
    )
    audio_response: Optional[str] = Field(
        default=None,
        description="Audio response from the model.",
    )


class OmniModel:
    def __init__(
        self, openai_api_key: Optional[str] = None, ai_config: Optional[AIConfig] = None
    ):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set in the environment! "
                    "Set it for these integration tests."
                )
        if ai_config is None:
            try:
                ai_config = AIConfig.load("ai_config.yaml")
            except FileNotFoundError:
                raise ValueError(
                    "ai_config.yaml file not found! "
                    "Set it for these integration tests."
                )
        self.openai_api_key = openai_api_key
        self.ai_config = ai_config
        self.text_model = TextModel(ai_config=ai_config, openai_api_key=openai_api_key)
        self.vision_model = VisionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.image_generation_model = ImageGenerationModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.audio_recognition_model = AudioRecognitionModel(
            ai_config=ai_config, openai_api_key=openai_api_key
        )
        self.moderation_model = Moderation(
            ai_config=ai_config, openai_api_key=openai_api_key
        )

    def run(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages_history: Optional[List[Dict[str, str]]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> OmniModelOutput:
        """
        Run the OmniModel with the provided inputs and return the output.
        """

        image_description = None
        if in_memory_image_stream is not None:
            image_description = self.vision_model.run(
                in_memory_image_stream=in_memory_image_stream,
            )

        audio_description = None
        if in_memory_audio_stream is not None:
            audio_description = self.audio_recognition_model.run(
                in_memory_audio_stream=in_memory_audio_stream,
            )

        input_data = OmniModelInput(
            system_prompt=system_prompt,
            user_input=user_input,
            messages_history=messages_history,
            image_description=image_description,
            audio_description=audio_description,
        )

        # Determine the output type based on the input data
        output_type = self.text_model.run(
            system_prompt=input_data.system_prompt,
            pydantic_model=OmniModelOutputType,
            user_input=input_data.user_input,
        )

        # Process the input data based on the output type
        if output_type == "text":
            text_response = self.text_model.run(
                system_prompt=input_data.system_prompt,
                pydantic_model=OmniModelOutputType,
                user_input=input_data.user_input,
            )
            return OmniModelOutput(text_response=text_response)
        elif output_type == "image":
            image_response = self.image_generation_model.run(
                system_prompt=input_data.system_prompt,
                pydantic_model=OmniModelOutputType,
                in_memory_image_stream=input_data.in_memory_image_stream,
            )
            return OmniModelOutput(image_response=image_response)
        elif output_type == "audio":
            audio_response = self.audio_recognition_model.run(
                system_prompt=input_data.system_prompt,
                pydantic_model=OmniModelOutputType,
                in_memory_audio_stream=input_data.in_memory_audio_stream,
            )
            return OmniModelOutput(audio_response=audio_response)
        elif output_type == "text_with_image":
            text_response = self.text_model.run(
                system_prompt=input_data.system_prompt,
                pydantic_model=OmniModelOutputType,
                user_input=input_data.user_input,
            )
            image_response = self.image_generation_model.run(
                system_prompt=input_data.system_prompt,
                pydantic_model=OmniModelOutputType,
                in_memory_image_stream=input_data.in_memory_image_stream,
            )
            return OmniModelOutput(
                text_response=text_response, image_response=image_response
            )
        else:
            raise ValueError(
                f"Unsupported output type: {output_type}. Supported types are: text, image, audio, text_with_image."
            )
