import os
import pytest
from pydantic import BaseModel
from omnimodkit.ai_config import (
    AIConfig,
    Rate,
    TextGeneration,
    Model,
)
from omnimodkit.models_toolkit import ModelsToolkit


# Dummy Pydantic model to simulate structured output
class DummyResponseModel(BaseModel):
    message: str


@pytest.fixture
def ai_config():
    # Create a Rate instance with dummy pricing values.
    rate = Rate(
        input_token_price=0.0,
        output_token_price=0.0,
        input_pixel_price=0.0,
        output_pixel_price=0.0,
        input_audio_second_price=0.0,
        output_audio_second_price=0.0,
    )
    # Set up the TextGeneration config with gpt-4o as the default text model.
    text_generation = TextGeneration(
        moderation_needed=False,
        models={
            "gpt-4o": Model(
                name="gpt-4o",
                temperature=0.7,
                structured_output_max_tokens=100,
                request_timeout=60,
                is_default=True,
                rate=rate,
            )
        },
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return AIConfig(
        TextGeneration=text_generation,
    )


@pytest.fixture
def real_toolkit(ai_config) -> ModelsToolkit:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in the environment! "
            "Set it for these integration tests."
        )
    return ModelsToolkit(openai_api_key=openai_key, ai_config=ai_config)


@pytest.mark.integration
def test_get_text_response_integration(real_toolkit: ModelsToolkit):
    prompt = "Provide a short JSON object with a single key 'message' saying Hello"
    response = real_toolkit.get_text_response(prompt)
    assert isinstance(response, DummyResponseModel)
    # The model might not respond exactly with "Hello", so be flexible
    # or test the shape, e.g.:
    assert hasattr(response, "message")
    assert len(response.message) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_aget_get_text_response_integration(real_toolkit: ModelsToolkit):
    prompt = (
        "Provide a short JSON object with a single key 'message' saying Hi from async"
    )
    response = await real_toolkit.aget_get_text_response(prompt)
    assert isinstance(response, DummyResponseModel)
    assert len(response.message) > 0


@pytest.mark.integration
def test_stream_text_response_integration(real_toolkit: ModelsToolkit):
    prompt = "Stream a short JSON object with a single key 'message'"
    # We expect the streaming generator to return multiple tokens or chunked responses.
    responses = list(real_toolkit.stream_text_response(prompt))
    assert all(isinstance(resp, BaseModel) for resp in responses)
    # At least one chunk:
    assert len(responses) > 0
    assert hasattr(responses[0], "message")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_astream_text_response_integration(real_toolkit: ModelsToolkit):
    prompt = "Stream (async) a short JSON object with a single key 'message'"
    responses = []
    async for res in real_toolkit.astream_text_response(prompt):
        responses.append(res)
    assert len(responses) > 0
    assert isinstance(responses[0], BaseModel)
    assert hasattr(responses[0], "message")
