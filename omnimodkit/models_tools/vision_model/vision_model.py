from ..base_model_toolkit import BaseModelToolkit


class VisionModel(BaseModelToolkit):
    def __init__(self, model):
        self.client = model
