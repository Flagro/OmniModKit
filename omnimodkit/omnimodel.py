import io
import asyncio
from typing import Optional, Union, Generator, AsyncGenerator, List, Literal
from pydantic import BaseModel, Field, ConfigDict, create_model
from .ai_config import AIConfig
from .models_toolkit import ModelsToolkit
from .base_toolkit_model import OpenAIMessage


AvailableModelType = Literal[
    "text", "vision", "image_generation", "audio_recognition", "audio_generation"
]


class TextResponse(BaseModel):
    text: str = Field(
        default="",
        description="Text response generated by the model.",
    )


class TextStreamingResponse(BaseModel):
    text_response: bool = Field(
        default=False,
        description="Indicates if the response is a text response.",
    )


class ImageResponse(BaseModel):
    image_description_to_generate: str = Field(
        default="",
        description="Description of the generated image.",
    )


class AudioResponse(BaseModel):
    audio_description_to_generate: str = Field(
        default="",
        description="Description of the generated audio.",
    )


class TextWithImageResponse(BaseModel):
    text: str = Field(
        default="",
        description=(
            "Text response generated by the model. "
            "When generating the text for this field, assume "
            "that the image with given description has already been generated "
            "and it will be attached to the response as an attachment."
        ),
    )
    image_description_to_generate: str = Field(
        default="",
        description="Description of the generated image.",
    )


class TextWithImageStreamingResponse(BaseModel):
    text_generation_needed: bool = Field(
        default=False,
        description="Indicates if text generation is needed.",
    )
    image_description_to_generate: str = Field(
        default="",
        description="Description of the generated image.",
    )


class OmniModelOutput(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    total_text: Optional[str] = Field(
        default=None,
        description="Text response from the model.",
    )
    text_new_chunk: Optional[str] = Field(
        default=None,
        description="New chunk of text response from the model.",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Generated image URL response from the model.",
    )
    audio_bytes: Optional[io.BytesIO] = Field(
        default=None,
        description="In-memory audio bytes response from the model.",
    )
    total_price: Optional[float] = Field(
        default=None,
        description="Total price of the model response based on input and output.",
    )


class OmniModel:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        ai_config: Optional[AIConfig] = None,
        allowed_models: Optional[List[AvailableModelType]] = None,
        allow_default_ai_config: bool = False,
    ):
        self.ai_config = ai_config
        self.modkit = ModelsToolkit(
            openai_api_key=openai_api_key,
            ai_config=ai_config,
            allow_default_ai_config=allow_default_ai_config,
        )
        self.allowed_models = allowed_models

    def _all_models_allowed(self) -> bool:
        """
        Check if all model types are allowed.
        """
        return self.allowed_models is None

    def _can_use_model(self, model_type: AvailableModelType) -> bool:
        """
        Check if a specific model type is allowed.
        """
        return self._all_models_allowed() or model_type in self.allowed_models

    def _get_allowed_output_types(self, is_streaming: bool = False) -> List[type]:
        """
        Get the list of allowed output type classes based on allowed_models.
        """
        allowed_types = []

        if self._can_use_model("text"):
            if is_streaming:
                allowed_types.append(TextStreamingResponse)
            else:
                allowed_types.append(TextResponse)

        if self._can_use_model("image_generation"):
            allowed_types.append(ImageResponse)

        if self._can_use_model("audio_generation"):
            allowed_types.append(AudioResponse)

        if self._can_use_model("text") and self._can_use_model("image_generation"):
            if is_streaming:
                allowed_types.append(TextWithImageStreamingResponse)
            else:
                allowed_types.append(TextWithImageResponse)

        return allowed_types

    def _create_dynamic_output_type_model(self, is_streaming: bool = False) -> type:
        """
        Create a dynamic output type model based on allowed models.
        """
        allowed_types = self._get_allowed_output_types(is_streaming=is_streaming)

        if is_streaming:
            union_types = allowed_types or [TextStreamingResponse]
        else:
            union_types = allowed_types or [TextResponse]

        if len(union_types) == 1:
            union_type = union_types[0]
        else:
            union_type = Union[tuple(union_types)]

        DynamicOmniModelOutputType = create_model(
            "DynamicOmniModelOutputType",
            output_type=(
                union_type,
                Field(description="Type of output expected from the model."),
            ),
        )

        return DynamicOmniModelOutputType

    def _compose_user_input(
        self,
        user_input: Optional[str] = None,
        image_description: Optional[str] = None,
        audio_description: Optional[str] = None,
    ) -> str:
        """
        Compose the user input by combining text, image description, and audio description.
        """
        if user_input is None:
            user_input = ""

        if image_description is not None:
            user_input += f" {image_description}"

        if audio_description is not None:
            user_input += f" {audio_description}"

        return user_input.strip()

    def _get_user_input(
        self,
        user_input: Optional[str] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> str:
        image_description = None
        if in_memory_image_stream is not None and self._can_use_model("vision"):
            image_description_object = self.modkit.vision_model.run_default(
                in_memory_image_stream=in_memory_image_stream,
            )
            image_description = str(image_description_object)

        audio_description = None
        if in_memory_audio_stream is not None and self._can_use_model(
            "audio_recognition"
        ):
            audio_description_object = self.modkit.audio_recognition_model.run_default(
                in_memory_audio_stream=in_memory_audio_stream,
            )
            audio_description = str(audio_description_object)

        return self._compose_user_input(
            user_input=user_input,
            image_description=image_description,
            audio_description=audio_description,
        )

    async def _aget_user_input(
        self,
        user_input: Optional[str] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> str:
        image_description = None
        if in_memory_image_stream is not None and self._can_use_model("vision"):
            image_description_object = await self.modkit.vision_model.arun_default(
                in_memory_image_stream=in_memory_image_stream,
            )
            image_description = str(image_description_object)
        audio_description = None
        if in_memory_audio_stream is not None and self._can_use_model(
            "audio_recognition"
        ):
            audio_description_object = (
                await self.modkit.audio_recognition_model.arun_default(
                    in_memory_audio_stream=in_memory_audio_stream,
                )
            )
            audio_description = str(audio_description_object)
        return self._compose_user_input(
            user_input=user_input,
            image_description=image_description,
            audio_description=audio_description,
        )

    def run(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> OmniModelOutput:
        """
        Run the OmniModel with the provided inputs and return the output.
        """
        user_input = self._get_user_input(
            user_input=user_input,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

        # Determine the output type based on the input data
        dynamic_output_type_model = self._create_dynamic_output_type_model(
            is_streaming=False
        )
        output_type_model = self.modkit.text_model.run(
            system_prompt=system_prompt,
            pydantic_model=dynamic_output_type_model,
            user_input=user_input,
            communication_history=communication_history,
        )
        output_type = output_type_model.output_type

        # Process the input data based on the output type
        if isinstance(output_type, ImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = self.modkit.image_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(image_url=image_response.image_url)
        elif isinstance(output_type, AudioResponse):
            if not self._can_use_model("audio_generation"):
                raise ValueError(
                    "Audio generation is not allowed with the current model configuration."
                )
            audio_response = self.modkit.audio_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.audio_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(audio_bytes=audio_response.audio_bytes)
        elif isinstance(output_type, TextWithImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = self.modkit.image_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(
                total_text=output_type.text,
                image_url=image_response.image_url,
            )
        elif isinstance(output_type, TextResponse):
            result = OmniModelOutput(total_text=output_type.text)
        else:
            raise ValueError("Unexpected output type received from the model.")

        return self.inject_price(
            output=result,
            user_input=user_input,
            system_prompt=system_prompt,
            communication_history=communication_history,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

    async def arun(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> OmniModelOutput:
        """
        Asynchronously run the OmniModel with the provided inputs and return the output.
        """
        user_input = await self._aget_user_input(
            user_input=user_input,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

        # Determine the output type based on the input data
        dynamic_output_type_model = self._create_dynamic_output_type_model(
            is_streaming=False
        )
        output_type_model = await self.modkit.text_model.arun(
            system_prompt=system_prompt,
            pydantic_model=dynamic_output_type_model,
            user_input=user_input,
            communication_history=communication_history,
        )
        output_type = output_type_model.output_type

        # Process the input data based on the output type
        if isinstance(output_type, ImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = await self.modkit.image_generation_model.arun_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(image_url=image_response.image_url)
        elif isinstance(output_type, AudioResponse):
            if not self._can_use_model("audio_generation"):
                raise ValueError(
                    "Audio generation is not allowed with the current model configuration."
                )
            audio_response = await self.modkit.audio_generation_model.arun_default(
                system_prompt=system_prompt,
                user_input=output_type.audio_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(audio_bytes=audio_response.audio_bytes)
        elif isinstance(output_type, TextWithImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = await self.modkit.image_generation_model.arun_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            result = OmniModelOutput(
                total_text=output_type.text,
                image_url=image_response.image_url,
            )
        elif isinstance(output_type, TextResponse):
            result = OmniModelOutput(total_text=output_type.text)
        else:
            raise ValueError("Unexpected output type received from the model.")
        return self.inject_price(
            output=result,
            user_input=user_input,
            system_prompt=system_prompt,
            communication_history=communication_history,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

    def stream(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> Generator[OmniModelOutput, None, None]:
        """
        Run the OmniModel with the provided inputs and return the output.
        """
        user_input = self._get_user_input(
            user_input=user_input,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

        # Determine the output type based on the input data
        dynamic_output_type_model = self._create_dynamic_output_type_model(
            is_streaming=True
        )
        output_type_model = self.modkit.text_model.run(
            system_prompt=system_prompt,
            pydantic_model=dynamic_output_type_model,
            user_input=user_input,
            communication_history=communication_history,
        )
        output_type = output_type_model.output_type

        # Process the input data based on the output type
        if isinstance(output_type, ImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = self.modkit.image_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            final_response = OmniModelOutput(image_url=image_response.image_url)
        elif isinstance(output_type, AudioResponse):
            if not self._can_use_model("audio_generation"):
                raise ValueError(
                    "Audio generation is not allowed with the current model configuration."
                )
            audio_response = self.modkit.audio_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.audio_description_to_generate,
                communication_history=communication_history,
            )
            final_response = OmniModelOutput(audio_bytes=audio_response.audio_bytes)
        elif isinstance(output_type, TextWithImageStreamingResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = self.modkit.image_generation_model.run_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            adjusted_text_generation_system_prompt = (
                f"{system_prompt}\n"
                "Generate text assuming that the image with the following description has "
                "already been generated and it will be attached to the response as an attachment:"
                f"\n{output_type.image_description_to_generate}"
            )
            text_response = self.modkit.text_model.stream_default(
                system_prompt=adjusted_text_generation_system_prompt,
                user_input=user_input,
                communication_history=communication_history,
            )
            # No point in streaming text with image synchronously,
            # as the image generation is a long-running task
            # (longer than text generation).
            final_response = OmniModelOutput(
                total_text=text_response,
                image_url=image_response.image_url,
            )
        elif isinstance(output_type, TextStreamingResponse):
            total_text = ""
            for chunk in self.modkit.text_model.stream_default(
                system_prompt=system_prompt,
                user_input=user_input,
                communication_history=communication_history,
            ):
                total_text += chunk.text_chunk
                yield OmniModelOutput(
                    total_text=total_text,
                    text_new_chunk=chunk.text_chunk,
                )
            final_response = OmniModelOutput(total_text=total_text, text_new_chunk="")
        else:
            raise ValueError("Unexpected output type received from the model.")
        return self.inject_price(
            output=final_response,
            user_input=user_input,
            system_prompt=system_prompt,
            communication_history=communication_history,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

    async def astream(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> AsyncGenerator[OmniModelOutput, None]:
        """
        Asynchronously run the OmniModel with the provided inputs and return the output.
        """
        user_input = await self._aget_user_input(
            user_input=user_input,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

        # Determine the output type based on the input data
        dynamic_output_type_model = self._create_dynamic_output_type_model(
            is_streaming=True
        )
        output_type_model = await self.modkit.text_model.arun(
            system_prompt=system_prompt,
            pydantic_model=dynamic_output_type_model,
            user_input=user_input,
            communication_history=communication_history,
        )
        output_type = output_type_model.output_type

        # Process the input data based on the output type
        if isinstance(output_type, ImageResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response = await self.modkit.image_generation_model.arun_default(
                system_prompt=system_prompt,
                user_input=output_type.image_description_to_generate,
                communication_history=communication_history,
            )
            final_response = OmniModelOutput(image_url=image_response.image_url)
        elif isinstance(output_type, AudioResponse):
            if not self._can_use_model("audio_generation"):
                raise ValueError(
                    "Audio generation is not allowed with the current model configuration."
                )
            audio_response = await self.modkit.audio_generation_model.arun_default(
                system_prompt=system_prompt,
                user_input=output_type.audio_description_to_generate,
                communication_history=communication_history,
            )
            final_response = OmniModelOutput(audio_bytes=audio_response.audio_bytes)
        elif isinstance(output_type, TextWithImageStreamingResponse):
            if not self._can_use_model("image_generation"):
                raise ValueError(
                    "Image generation is not allowed with the current model configuration."
                )
            image_response_task = asyncio.create_task(
                self.modkit.image_generation_model.arun_default(
                    system_prompt=system_prompt,
                    user_input=output_type.image_description_to_generate,
                    communication_history=communication_history,
                )
            )
            adjusted_text_generation_system_prompt = (
                f"{system_prompt}\n"
                "Generate text assuming that the image with the following description has "
                "already been generated and it will be attached to the response as an attachment:"
                f"\n{output_type.image_description_to_generate}"
            )
            total_text = ""
            async for chunk in self.modkit.text_model.astream_default(
                system_prompt=adjusted_text_generation_system_prompt,
                user_input=user_input,
                communication_history=communication_history,
            ):
                if chunk.text_chunk:
                    total_text += chunk.text_chunk
                    yield OmniModelOutput(
                        total_text=total_text,
                        text_new_chunk=chunk.text_chunk,
                    )
            image_response = await image_response_task
            final_response = OmniModelOutput(
                total_text=total_text,
                image_url=image_response.image_url,
            )
        elif isinstance(output_type, TextStreamingResponse):
            total_text = ""
            async for chunk in self.modkit.text_model.astream_default(
                system_prompt=system_prompt,
                user_input=user_input,
                communication_history=communication_history,
            ):
                total_text += chunk.text_chunk
                yield OmniModelOutput(
                    total_text=total_text,
                    text_new_chunk=chunk.text_chunk,
                )
            final_response = OmniModelOutput(total_text=total_text, text_new_chunk="")
        else:
            raise ValueError("Unexpected output type received from the model.")
        yield self.inject_price(
            output=final_response,
            user_input=user_input,
            system_prompt=system_prompt,
            communication_history=communication_history,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )

    def get_price(
        self,
        output: OmniModelOutput,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> float:
        """
        Get the price of the model based on input and output text, image, and audio.
        """
        total_input_text = (
            (user_input or "")
            + (system_prompt or "")
            + "\n".join([msg["text"] for msg in communication_history or []])
        )

        # Only calculate prices for allowed models
        input_image = in_memory_image_stream if self._can_use_model("vision") else None
        output_image_url = (
            output.image_url if self._can_use_model("image_generation") else None
        )
        input_audio = (
            in_memory_audio_stream if self._can_use_model("audio_recognition") else None
        )
        output_audio = (
            output.audio_bytes if self._can_use_model("audio_generation") else None
        )

        return self.modkit.get_price(
            input_text=total_input_text,
            output_text=output.total_text,
            input_image=input_image,
            output_image_url=output_image_url,
            input_audio=input_audio,
            output_audio=output_audio,
        )

    def estimate_price(
        self,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> float:
        """
        Estimate the price of the model based on input and output text, image, and audio.
        """
        total_input_text = (
            (user_input or "")
            + (system_prompt or "")
            + "\n".join([msg["text"] for msg in communication_history or []])
        )

        # Only calculate prices for allowed models
        input_image = in_memory_image_stream if self._can_use_model("vision") else None
        output_image_url = (
            None if not self._can_use_model("image_generation") else "dummy_image_url"
        )
        input_audio = (
            in_memory_audio_stream if self._can_use_model("audio_recognition") else None
        )
        output_audio = (
            None
            if not self._can_use_model("audio_generation")
            else io.BytesIO(b"dummy_audio")
        )

        return self.modkit.get_price(
            input_text=total_input_text,
            output_text=None,
            input_image=input_image,
            output_image_url=output_image_url,
            input_audio=input_audio,
            output_audio=output_audio,
        )

    def inject_price(
        self,
        output: OmniModelOutput,
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        communication_history: Optional[List[OpenAIMessage]] = None,
        in_memory_image_stream: Optional[io.BytesIO] = None,
        in_memory_audio_stream: Optional[io.BytesIO] = None,
    ) -> OmniModelOutput:
        """
        Inject the price into the OmniModelOutput based on the input and output.
        """
        output.total_price = self.get_price(
            output=output,
            user_input=user_input,
            system_prompt=system_prompt,
            communication_history=communication_history,
            in_memory_image_stream=in_memory_image_stream,
            in_memory_audio_stream=in_memory_audio_stream,
        )
        return output
