import os
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from omnimodkit.models_toolkit import ModelsToolkit


# Dummy Pydantic model to simulate structured output
class DummyResponseModel(BaseModel):
    message: str


@pytest.fixture
def mock_ai_config():
    mock_config = MagicMock()
    mock_config.TextGeneration.Models = {
        "gpt-test": MagicMock(
            name="gpt-test",
            temperature=0.7,
            default=True,
            structured_output_max_tokens=100,
        )
    }
    mock_config.TextGeneration.moderation_needed = False
    return mock_config


@pytest.fixture
def real_toolkit(mock_ai_config):
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in the environment! "
            "Set it for these integration tests."
        )
    return ModelsToolkit(openai_api_key=openai_key, ai_config=mock_ai_config)


@pytest.mark.integration
def test_get_text_response_integration(real_toolkit):
    prompt = "Provide a short JSON object with a single key 'message' saying Hello"
    response = real_toolkit.get_text_response(prompt)
    assert isinstance(response, DummyResponseModel)
    # The model might not respond exactly with "Hello", so be flexible
    # or test the shape, e.g.:
    assert hasattr(response, "message")
    assert len(response.message) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_aget_get_text_response_integration(real_toolkit):
    prompt = (
        "Provide a short JSON object with a single key 'message' saying Hi from async"
    )
    response = await real_toolkit.aget_get_text_response(prompt)
    assert isinstance(response, DummyResponseModel)
    assert len(response.message) > 0


@pytest.mark.integration
def test_stream_text_response_integration(real_toolkit):
    prompt = "Stream a short JSON object with a single key 'message'"
    # We expect the streaming generator to return multiple tokens or chunked responses.
    responses = list(real_toolkit.stream_text_response(prompt))
    assert all(isinstance(resp, DummyResponseModel) for resp in responses)
    # At least one chunk:
    assert len(responses) > 0
    assert hasattr(responses[0], "message")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_astream_text_response_integration(real_toolkit):
    prompt = "Stream (async) a short JSON object with a single key 'message'"
    responses = []
    async for res in real_toolkit.astream_text_response(prompt):
        responses.append(res)
    assert len(responses) > 0
    assert isinstance(responses[0], DummyResponseModel)
    assert hasattr(responses[0], "message")
