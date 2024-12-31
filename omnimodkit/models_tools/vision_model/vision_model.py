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


class VisionModel(BaseModelToolkit):
    model_name = "vision"

    def __init__(self, openai_api_key: str):
        self.vision_model = OpenAI(
            api_key=openai_api_key,
            model=self.get_model().name,
        )

    def get_models_dict(self) -> Dict[str, Model]:
        return self.ai_config.TextGeneration.Models

    def get_default_attr(self) -> str:
        return "vision_default"

    def run(
        self,
        in_memory_image_stream: io.BytesIO,
        pydantic_object: Optional[Type[BaseModel]] = None,
    ) -> BaseModel:
        if pydantic_object is None:
            pydantic_object = PromptManager.get_default_image_information()
        # Encode in base64:
        image_base64 = base64.b64encode(in_memory_image_stream.getvalue()).decode()
        parser = JsonOutputParser(pydantic_object=pydantic_object)

        model = ChatOpenAI(
            temperature=0.7, model=self.get_model().name, max_tokens=1024
        )

        prompt = """
            Based on the image, fill out the provided fields.
        """

        msg = model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": parser.get_format_instructions()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
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
        image_pixels_count: int,
    ) -> float:
        input_pixel_price = self.get_model().rate.input_pixel_price
        return image_pixels_count * input_pixel_price
