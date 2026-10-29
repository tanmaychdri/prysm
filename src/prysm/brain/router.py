"""
Router LLM Provider for PRYSM.

Manages local models (Ollama) as primary intelligence and cloud API models as secondary harness.
Routes queries to:
- Primary Local Model (qwen2.5-coder:14b): Default for general assistant tasks, coding, OS tools, and system commands.
- Thinking Local Model (qwen3.5:9b): For analytical, reasoning, planning, and deep thinking questions.
- Secondary API Harness (e.g. Groq / OpenAI): Secondary fallback or when the user explicitly switches to API models.
"""

import logging
import re
from typing import Any

from prysm.brain.provider import LLMProvider
from prysm.brain.providers.openai_provider import OpenAILLMProvider
from prysm.config.settings import Settings
from prysm.models.interactions import BrainResponse, LLMMessage

logger = logging.getLogger(__name__)

# Thinking triggers and patterns
_THINKING_KEYWORDS = {
    "think", "reason", "ponder", "reflect", "hypothesize", "philosophize",
    "analyze", "analysis", "evaluate", "evaluation", "assessment",
    "pros and cons", "tradeoff", "tradeoffs", "trade-off", "trade-offs",
    "implication", "implications", "step by step", "deep dive", "puzzle",
    "riddle", "paradox", "logic", "logical", "deduce", "deduction", "probability"
}

_THINKING_PHRASES = [
    r"\bthink\s+(?:about|deeply|carefully|through)\b",
    r"\bwhy\s+(?:is|are|does|do|did|would|should|could|was|were)\b",
    r"\bhow\s+(?:come|does|would|should|could)\b",
    r"\bwhat\s+(?:would happen if|is the difference between|are the trade-?offs?|are the pros and cons)\b",
    r"\btrade-?offs?\b",
    r"\bpros\s+and\s+cons\b",
    r"\bdifference\s+between\b",
    r"\blet'?s\s+think\b",
    r"\bstep\s+by\s+step\b",
    r"\bexplain\s+(?:the\s+logic|why|how)\b",
    r"\bcompare\s+and\s+contrast\b",
    r"\bsolve\s+(?:this\s+)?(?:riddle|puzzle|problem)\b",
    r"\bphilosoph(?:y|ical)\b",
    r"\[think\]",
    r"#think",
]
_THINKING_REGEX = re.compile("|".join(_THINKING_PHRASES), re.IGNORECASE)

# Coding and OS actions that should stay on the primary coder model
_CODING_OS_PHRASES = [
    r"\b(?:write|code|create|build|implement|debug|fix|refactor|optimize)\s+(?:a|an|the|this)?\s*(?:python|script|function|class|code|program|test|regex|sql|query|html|css|bash|powershell|api)\b",
    r"\b(?:open|launch|start|close|kill|run|execute)\s+(?:app|application|chrome|spotify|notepad|calc|terminal|powershell)\b",
    r"\b(?:set|turn|change|mute)\s+(?:the\s+)?volume\b",
    r"\b(?:shut\s*down|restart|reboot|sleep|lock\s+pc|battery|cpu|ram|memory\s+usage|system\s+info)\b",
    r"\bclipboard\b",
]
_CODING_OS_REGEX = re.compile("|".join(_CODING_OS_PHRASES), re.IGNORECASE)


def is_thinking_query(messages: list[LLMMessage]) -> bool:
    """Determine whether the latest user prompt requires the thinking model."""
    last_user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user_msg or not last_user_msg.content:
        return False

    text = last_user_msg.content.strip().lower()

    # If it's explicitly a coding or OS tool command, prioritize coder primary model
    if _CODING_OS_REGEX.search(text) and not re.search(r"\[think\]|#think|\bthink\s+(?:deeply|carefully)\b", text):
        return False

    if _THINKING_REGEX.search(text):
        return True

    words = set(re.findall(r"\b[a-z]{3,}\b", text))
    return bool(words & _THINKING_KEYWORDS and ("?" in text or len(words) > 6))


class RouterLLMProvider(LLMProvider):
    """
    Intelligent router provider that manages local Ollama models (Coder primary +
    Thinking specialist) and cloud API models as secondary fallback harness.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.llm_mode.lower()  # "local" or "api"

        # Local primary model (qwen2.5-coder:14b)
        self.primary_provider = OpenAILLMProvider(
            api_key="ollama",
            model=settings.local_primary_model,
            base_url=settings.ollama_base_url,
        )

        # Local thinking model (qwen3.5:9b)
        self.thinking_provider = OpenAILLMProvider(
            api_key="ollama",
            model=settings.local_thinking_model,
            base_url=settings.ollama_base_url,
        )

        # Secondary API harness (e.g. Groq / OpenAI)
        self.api_provider = OpenAILLMProvider(
            api_key=settings.llm_api_key or "sk-dummy",
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )

        self.last_used_model: str = self.settings.local_primary_model
        logger.info(f"RouterLLMProvider initialized in {self.get_status_description()}")

    def set_mode(self, mode: str) -> None:
        """Switch between 'local' and 'api' model modes."""
        mode_clean = mode.strip().lower()
        if mode_clean in ("local", "offline", "ollama"):
            self.mode = "local"
        elif mode_clean in ("api", "cloud", "online", "secondary"):
            self.mode = "api"
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'local' or 'api'.")
        logger.info(f"Model routing mode switched to: {self.mode}")

    def get_mode(self) -> str:
        return self.mode

    def get_status_description(self) -> str:
        if self.mode == "local":
            return (
                f"Local Mode [Primary: {self.settings.local_primary_model}, "
                f"Thinking: {self.settings.local_thinking_model}] (Secondary API Harness: {self.settings.llm_model})"
            )
        else:
            return f"API Mode [Active Harness: {self.settings.llm_model}] (Local Models Standby)"

    async def generate_response(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> BrainResponse:
        """
        Route response generation to the appropriate model based on current mode,
        query type (thinking vs primary), with automatic fallback.
        """
        if self.mode == "local":
            thinking = is_thinking_query(messages)
            target_provider = self.thinking_provider if thinking else self.primary_provider
            model_name = (
                self.settings.local_thinking_model if thinking else self.settings.local_primary_model
            )
            role_desc = "thinking" if thinking else "coder primary"

            logger.info(f"Routing to local {role_desc} model: {model_name}")
            try:
                response = await target_provider.generate_response(messages, tools)
                self.last_used_model = model_name
                return response
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Local {role_desc} model ({model_name}) failed: {e}. "
                    f"Falling back to secondary API harness ({self.settings.llm_model})."
                )
                response = await self.api_provider.generate_response(messages, tools)
                self.last_used_model = f"{self.settings.llm_model} (fallback)"
                return response
        else:
            logger.info(f"Routing to secondary API harness: {self.settings.llm_model}")
            try:
                response = await self.api_provider.generate_response(messages, tools)
                self.last_used_model = self.settings.llm_model
                return response
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"API harness ({self.settings.llm_model}) failed: {e}. "
                    f"Falling back to local primary model ({self.settings.local_primary_model})."
                )
                response = await self.primary_provider.generate_response(messages, tools)
                self.last_used_model = f"{self.settings.local_primary_model} (fallback)"
                return response
