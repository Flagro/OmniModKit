from typing import Dict, Optional
import yaml
from pydantic import BaseModel


class Rate(BaseModel):
    input_token_price: Optional[float]
    output_token_price: Optional[float]
    input_pixel_price: Optional[float]
    output_pixel_price: Optional[float]
    input_audio_second_price: Optional[float]
    output_audio_second_price: Optional[float]


class Model(BaseModel):
    name: str
    temperature: float
    structured_output_max_tokens: int = 1024
    request_timeout: float = 60
    default: Optional[bool] = False
    rate: Rate


class TextGeneration(BaseModel):
    moderation_needed: bool = True
    max_tokens: int
    top_p: int
    frequency_penalty: int
    presence_penalty: int
    Models: Dict[str, Model]


class ImageGeneration(BaseModel):
    moderation_needed: bool = True
    output_image_size: str
    Models: Dict[str, Model]


class AudioRecognition(BaseModel):
    moderation_needed: bool = True
    Models: Dict[str, Model]


class Vision(BaseModel):
    moderation_needed: bool = True
    Models: Dict[str, Model]


class AIConfig(BaseModel):
    TextGeneration: TextGeneration
    ImageGeneration: ImageGeneration
    AudioRecognition: AudioRecognition
    Vision: Vision

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
            if cls.__name__ in config_dict:
                return cls(**config_dict[cls.__name__])
            else:
                raise KeyError(f"{cls.__name__} not found in {file_path}")
