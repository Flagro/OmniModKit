from typing import List

from .base_tool import BaseTool
from ..models_toolkit import ModelsToolkit
from ..prompt_manager import PromptManager


class AIAgentToolkit:
    def __init__(
        self,
        models_toolkit: ModelsToolkit,
        prompt_manager: PromptManager,
        tools: List[BaseTool],
    ):
        self.models_toolkit = models_toolkit
        self.prompt_manager = prompt_manager
        self.tools = tools
