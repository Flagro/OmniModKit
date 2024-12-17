from .autofact_generation import CheckIfFactsNeededTool, ComposeFactsBasedOnMessagesTool
from .check_engage_needed import CheckEngageNeededTool
from .generate_image import ImageGeneratorTool
from .get_facts import GetChatFactsTool, GetUserFactsTool
from .get_response import GetResponseTool
from ..models_toolkit import ModelsToolkit
from ...prompt_manager import PromptManager


class AIAgentToolkit:
    def __init__(
        self,
        models_toolkit: ModelsToolkit,
        prompt_manager: PromptManager,
    ):
        self.check_if_facts_needed = CheckIfFactsNeededTool(
            models_toolkit, prompt_manager
        )
        self.compose_facts_based_on_messages = ComposeFactsBasedOnMessagesTool(
            models_toolkit, prompt_manager
        )
        self.check_engage_needed = CheckEngageNeededTool(
            models_toolkit, prompt_manager
        )
        self.image_generator = ImageGeneratorTool(
            models_toolkit, prompt_manager
        )
        self.get_chat_facts = GetChatFactsTool(
            models_toolkit, prompt_manager
        )
        self.get_user_facts = GetUserFactsTool(
            models_toolkit, prompt_manager
        )
        self.get_response = GetResponseTool(
            models_toolkit, prompt_manager
        )
