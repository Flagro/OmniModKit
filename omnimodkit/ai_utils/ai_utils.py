from .describe_image import DescribeImageUtililty
from ..models_toolkit import ModelsToolkit


class AgentUtils:
    def __init__(
        self,
        models_toolkit: ModelsToolkit,
    ):
        self.describe_image = DescribeImageUtililty(
            models_toolkit
        )
