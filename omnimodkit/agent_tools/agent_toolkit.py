from typing import List

from .base_tool import BaseTool
from ..models_toolkit import ModelsToolkit


class AIAgentToolkit:
    def __init__(
        self,
        models_toolkit: ModelsToolkit,
        tools: List[BaseTool],
    ):
        self.models_toolkit = models_toolkit
        self.tools = tools
