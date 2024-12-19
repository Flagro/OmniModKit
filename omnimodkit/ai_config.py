from typing import Dict, Optional
import yaml
from pydantic import BaseModel


class Rate(BaseModel):
    input_token_price: Optional[float]
    output_token_price: Optional[float]
    input_pixel_price: Optional[float]
    output_pixel_price: Optional[float]


class Model(BaseModel):
    name: str
    text_default: Optional[bool] = False
    vision_default: Optional[bool] = False
    image_generation_default: Optional[bool] = False
    rate: Rate


class TextGeneration(BaseModel):
    temperature: float
    max_tokens: int
    top_p: int
    frequency_penalty: int
    presence_penalty: int
    request_timeout: float
    Models: Dict[str, Model]


class ImageGeneration(BaseModel):
    output_image_size: str
    request_timeout: float
    Models: Dict[str, Model]


class AIConfig(BaseModel):
    TextGeneration: TextGeneration
    ImageGeneration: ImageGeneration

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
            if cls.__name__ in config_dict:
                return cls(**config_dict[cls.__name__])
            else:
                raise KeyError(f"{cls.__name__} not found in {file_path}")
