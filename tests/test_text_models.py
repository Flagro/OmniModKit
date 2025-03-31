import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from omnimodkit.models_toolkit import ModelsToolkit
from omnimodkit.text_model.text_model import TextModel


# Dummy Pydantic model to simulate structured output
class DummyResponseModel(BaseModel):
    message: str


@pytest.fixture
def mock_ai_config():
    # Minimal config mock with required interface
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
def toolkit(mock_ai_config):
    with patch(
        "your_module.text_model.text_model.TextModel.run",
        return_value=DummyResponseModel(message="Hello"),
    ):
        return ModelsToolkit(openai_api_key="fake-key", ai_config=mock_ai_config)


def test_get_text_response(toolkit):
    response = toolkit.get_text_response("Hello world. Reply only with 'Hello'")
    assert isinstance(response, DummyResponseModel)
    assert response.message == "Hello"


@pytest.mark.asyncio
async def test_aget_get_text_response(mock_ai_config):
    with patch.object(TextModel, "arun", new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = DummyResponseModel(message="Hi async")
        toolkit = ModelsToolkit(openai_api_key="fake-key", ai_config=mock_ai_config)
        response = await toolkit.aget_get_text_response(
            "Hello async. Reply only with 'Hi async'"
        )
        assert isinstance(response, DummyResponseModel)
        assert response.message == "Hi async"


def test_stream_text_response(toolkit):
    with patch.object(
        TextModel, "stream", return_value=iter([DummyResponseModel(message="Streamed")])
    ):
        responses = list(
            toolkit.stream_text_response("Stream it. Reply only with 'Streamed'")
        )
        assert all(isinstance(resp, DummyResponseModel) for resp in responses)
        assert responses[0].message == "Streamed"


@pytest.mark.asyncio
async def test_astream_text_response(mock_ai_config):
    async def async_gen():
        yield DummyResponseModel(message="Async stream")

    with patch.object(TextModel, "astream", return_value=async_gen()):
        toolkit = ModelsToolkit(openai_api_key="fake-key", ai_config=mock_ai_config)
        responses = []
        async for res in toolkit.astream_text_response(
            "Async stream test. Reply only with 'Async stream'"
        ):
            responses.append(res)
        assert responses
        assert isinstance(responses[0], DummyResponseModel)
        assert responses[0].message == "Async stream"
