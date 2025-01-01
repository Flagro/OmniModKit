import io
import base64
from typing import Type, Optional, Dict
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser

from ..base_model_toolkit import BaseModelToolkit
from ...ai_config import Model
from ...prompt_manager import PromptManager


class AudioRecognitionModel(BaseModelToolkit):
    model_name = "audio_recognition"
    default_attribute = "audio_recognition_default"

    def __init__(self, openai_api_key: str):
        self.audio_recognition_model = OpenAI(
            api_key=openai_api_key,
            model=self.get_model().name,
        )

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.AudioRecognition.Models

    def run(
        self,
        in_memory_audio_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        if pydantic_object is None:
            # TODO: prompt manager should be available as part of the utility
            pydantic_object = PromptManager.get_default_audio_information()
        # Encode in base64:
        audio_base64 = base64.b64encode(in_memory_audio_stream.getvalue()).decode()
        parser = JsonOutputParser(pydantic_object=pydantic_object)

        model = ChatOpenAI(
            temperature=0.7, model=self.get_model().name, max_tokens=1024
        )

        prompt = """
            Based on the audio, fill out the provided fields.
        """

        msg = model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": parser.get_format_instructions()},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": "mp3",
                            },
                        },
                    ]
                )
            ]
        )
        contents = msg.content
        parsed_output = parser.invoke(contents)
        return pydantic_object(**parsed_output)

    async def arun(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        # TODO: make it non-blocking
        return self.run(in_memory_image_stream, pydantic_object)

    def get_price(
        self,
        audio_length: int,
    ) -> float:
        input_token_price = self.get_model().rate.input_token_price
        return audio_length * input_token_price
