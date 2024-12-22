from ..base_model_toolkit import BaseModelToolkit


class AudioRecognitionModel(BaseModelToolkit):
    def __init__(self, model):
        self.client = model
