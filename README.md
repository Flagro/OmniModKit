# OmniModKit

Use convenient multimodal toolkit that operates with structured output.

Easily build agent tools on top of that.

# Implementation
This package utilizes the implemented langchain structured output pipelines.

# Installation

```bash
pip install omnimodkit
```

# Usage

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
```

# License
MIT license
