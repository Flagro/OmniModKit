# OmniModKit

Use convenient multimodal toolkit that operates with structured output.

Easily build agent tools on top of that.

# Implementation
This package utilizes the implemented langchain structured output pipelines.

# Installation

```bash
pip install omnimodkit
```

# Omnimodel Usage

- Import OmniModel
- Give it text/image/audio and get text/image/audio

```python
from omnimodkit import OmniModel

# Initialize the model
omni_model = OmniModel()

# Get image
omni_model.run(
    user_input="Give me an image of a cat",
)

# Get just text
omni_model.run(
    user_input="Tell me a joke",
)

# Get audio response
omni_model.run(
    user_input="Tell me a joke with voice",
)

# Get image and text
omni_model.run(
    user_input="Show me a cat and tell me about it",
)

# Stream responses
for response in omni_model.stream(
    user_input="Tell me a joke",
):
    print(response.text_new_chunk, end="|", flush=True)

# Async stream responses with image generation
last_response = None
async for response in omni_model.astream(
    user_input="Draw a cat and provide some text about it",
):
    if response.text_new_chunk:
        print(response.text_new_chunk, end="|", flush=True)
    last_response = response
print("\nFinal response:", last_response)

# Async stream responses
async for response in omni_model.astream(
    user_input="Tell me a joke",
):
    print(response.text_new_chunk, end="|", flush=True)

# Use audio recognition
import io
import requests

url = "https://cdn.openai.com/API/examples/data/ZyntriQix.wav"
audio_bytes = io.BytesIO(requests.get(url, timeout=10).content)
audio_bytes.name = "audio.wav"
omni_model.run(
    user_input="Draw an image based on the audio and tell me about it.",
    in_memory_audio_stream=audio_bytes,
)

# Use image recognition
import io
import requests

url = "https://raw.githubusercontent.com/Flagro/treefeeder/main/logo.png"
image_bytes = io.BytesIO(requests.get(url, timeout=10).content)
image_bytes.name = "image.png"
omni_model.run(
    user_input="Describe this image and generate a related image.",
    in_memory_image_stream=image_bytes,
)

# Estimate price for a model run
import io
import requests

url = "https://raw.githubusercontent.com/Flagro/treefeeder/main/logo.png"
image_bytes = io.BytesIO(requests.get(url, timeout=10).content)
image_bytes.name = "image.png"
omni_model.estimate_price(
    user_input="What is the capital of France?", in_memory_image_stream=image_bytes
)
```


# Modkit Usage

- Import ModelsToolkit
- Run appropriate models
- Get structured output response

```python
from omnimodkit import ModelsToolkit

# Initialize the model toolkit
modkit = ModelsToolkit()

# Run the model synchronously
modkit.text_model.run(
    user_input="What is the capital of France?",
)

# Stream responses from the model
for response in modkit.text_model.stream(
    user_input="What is the capital of France?",
):
    print(response, end="|", flush=True)

# Generate images
modkit.image_generation_model.run(
    user_input="Draw a cat",
)

# Use audio recognition
import io
import requests

url = "https://cdn.openai.com/API/examples/data/ZyntriQix.wav"
audio_bytes = io.BytesIO(requests.get(url, timeout=10).content)
audio_bytes.name = "audio.wav"
modkit.audio_recognition_model.run(
    in_memory_audio_stream=audio_bytes,
)

# Use image recognition
import io
import requests

url = "https://raw.githubusercontent.com/Flagro/treefeeder/main/logo.png"
image_bytes = io.BytesIO(requests.get(url, timeout=10).content)
image_bytes.name = "image.png"
modkit.vision_model.run(
    in_memory_image_stream=image_bytes,
)

# Use audio generation
modkit.audio_generation_model.run(
    user_input="Hello! How can I help you today?",
)
```

# License
MIT license
