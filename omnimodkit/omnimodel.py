import io
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict
from .ai_config import AIConfig
from .models_toolkit import ModelsToolkit


class OmniModelOutputType(BaseModel):
    output_type: Literal["text", "image", "audio", "text_with_image"] = Field(
        default="text",
        description="Type of output expected from the model.",
    )


class OmniModelOutput(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    text_response: Optional[str] = Field(
        default=None,
        description="Text response from the model.",
    )
    image_url_response: Optional[str] = Field(
        default=None,
        description="Generated image URL response from the model.",
    )
    audio_bytes_response: Optional[io.BytesIO] = Field(
        default=None,
        description="In-memory audio bytes response from the model.",
    )


class OmniModel:
    def __init__(
        self, openai_api_key: Optional[str] = None, ai_config: Optional[AIConfig] = None
    ):
        self.ai_config = ai_config
        self.modkit = ModelsToolkit(openai_api_key=openai_api_key, ai_config=ai_config)

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
            image_description_object = self.modkit.vision_model.run(
                in_memory_image_stream=in_memory_image_stream,
            )
            image_description = str(image_description_object)

        audio_description = None
        if in_memory_audio_stream is not None:
            audio_description_object = self.modkit.audio_recognition_model.run(
                in_memory_audio_stream=in_memory_audio_stream,
            )
            audio_description = str(audio_description_object)

        # Determine the output type based on the input data
        output_type = self.modkit.text_model.run(
            system_prompt=system_prompt,
            pydantic_model=OmniModelOutputType,
            user_input=user_input,
            messages_history=messages_history,
        )

        # Process the input data based on the output type
        if output_type == "text":
            text_response = self.modkit.text_model.run(
                system_prompt=system_prompt,
                user_input=user_input,
                messages_history=messages_history,
            )
            return OmniModelOutput(text_response=text_response.text)
        elif output_type == "image":
            image_description_response = self.modkit.text_model.run(
                system_prompt=system_prompt,
                user_input=user_input,
                messages_history=messages_history,
            )
            image_response = self.modkit.image_generation_model.run(
                system_prompt=system_prompt,
                user_input=image_description_response.text,
                messages_history=messages_history,
            )
            return OmniModelOutput(image_url_response=image_response.image_url)
        elif output_type == "audio":
            text_response = self.modkit.text_model.run(
                system_prompt=system_prompt,
                user_input=user_input,
                messages_history=messages_history,
            )
            audio_response = self.modkit.audio_generation_model.run(
                system_prompt=system_prompt,
                user_input=text_response.text,
                messages_history=messages_history,
            )
            return OmniModelOutput(audio_bytes_response=audio_response.audio_bytes)
        elif output_type == "text_with_image":
            text_response = self.modkit.text_model.run(
                system_prompt=system_prompt,
                user_input=user_input,
                messages_history=messages_history,
            )
            image_description_response = self.modkit.text_model.run(
                system_prompt=system_prompt,
                user_input=user_input,
                messages_history=messages_history,
            )
            image_response = self.modkit.image_generation_model.run(
                system_prompt=system_prompt,
                user_input=image_description_response.text,
                messages_history=messages_history,
            )
            return OmniModelOutput(
                text_response=text_response, image_response=image_response
            )
        else:
            raise ValueError(
                f"Unsupported output type: {output_type}. Supported types are: text, image, audio, text_with_image."
            )
