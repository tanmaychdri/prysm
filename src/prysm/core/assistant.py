import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prysm.brain.context import ContextManager
    from prysm.tools.executor import ToolExecutor

from prysm.brain.provider import LLMProvider
from prysm.core.events import (
    AssistantThinkingCompleted,
    AssistantThinkingStarted,
    ErrorOccurred,
    EventBus,
    InputReceived,
    ProcessingCompleted,
    ProcessingStarted,
    ResponseGenerated,
    StateChanged,
)
from prysm.core.exceptions import InvalidStateTransitionError
from prysm.core.lifecycle import Lifecycle
from prysm.core.state import AssistantState
from prysm.models.interactions import BrainResponse, UserInput
from prysm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8

# Valid state transitions
_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.STARTING: {AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.IDLE: {
        AssistantState.LISTENING,
        AssistantState.PROCESSING,
        AssistantState.STOPPING,
        AssistantState.ERROR,
    },
    AssistantState.LISTENING: {
        AssistantState.PROCESSING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.PROCESSING: {
        AssistantState.THINKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.THINKING: {
        AssistantState.EXECUTING_TOOL,
        AssistantState.RESPONDING,
        AssistantState.ERROR,
    },
    AssistantState.EXECUTING_TOOL: {
        AssistantState.THINKING,
        AssistantState.ERROR,
    },
    AssistantState.RESPONDING: {
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.ERROR: {AssistantState.IDLE, AssistantState.STOPPING},
    AssistantState.STOPPING: {AssistantState.STOPPED},
    AssistantState.STOPPED: set(),
}


class PrysmAssistant:
    """Core assistant — orchestrates the full request/response pipeline."""

    def __init__(
        self,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        llm_provider: LLMProvider,
        context_manager: "ContextManager",
        tool_executor: "ToolExecutor",
    ) -> None:
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.llm_provider = llm_provider
        self.context_manager = context_manager
        self.tool_executor = tool_executor

        self.state = AssistantState.STARTING
        self.lifecycle = Lifecycle()
        self.lifecycle.on_startup(self._initialize)
        self.lifecycle.on_shutdown(self._cleanup)

        self._stop_event: asyncio.Event = asyncio.Event()
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def set_state(self, new_state: AssistantState, reason: str | None = None) -> None:
        if self.state == new_state:
            return

        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed and new_state != AssistantState.ERROR:
            msg = f"Invalid transition {self.state.name} -> {new_state.name}"
            logger.warning(msg)
            raise InvalidStateTransitionError(msg)

        old_state = self.state
        self.state = new_state
        logger.debug(f"State: {old_state.name} -> {new_state.name}" + (f" ({reason})" if reason else ""))

        await self.event_bus.publish(
            StateChanged(previous_state=old_state, new_state=new_state, reason=reason)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        logger.info("PrysmAssistant initializing...")
        await self.set_state(AssistantState.IDLE, reason="startup complete")

    async def _cleanup(self) -> None:
        logger.info("PrysmAssistant shutting down...")
        if self.state not in (AssistantState.STOPPING, AssistantState.STOPPED):
            await self.set_state(AssistantState.STOPPING, reason="shutdown requested")

        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Save session and extract long-term memories before exit
        await self._persist_memory()

        await self.set_state(AssistantState.STOPPED, reason="cleanup complete")

    async def _persist_memory(self) -> None:
        """Save conversation history and extract long-term facts."""
        try:
            session_text = self.context_manager.format_session_as_text()
            if not session_text.strip():
                return

            # 1. Save raw conversation to disk
            self.context_manager.save_session()

            # 2. Extract new long-term facts via LLM
            if hasattr(self.context_manager, "_memory") and self.context_manager._memory:
                logger.info("Extracting long-term memories...")
                # Resolve the user's name so facts are personalized
                user_name = "the user"
                if (
                    hasattr(self.context_manager, "_session")
                    and self.context_manager._session
                    and self.context_manager._session.user
                ):
                    user_name = self.context_manager._session.user.display_name
                added = await self.context_manager._memory.extract_and_store(
                    session_text, self.llm_provider, user_name=user_name
                )
                if added:
                    logger.info(f"Stored {added} new memory facts")
        except Exception:
            logger.exception("Memory persistence failed")

    async def run(self) -> None:
        """Start the assistant and block until stopped."""
        await self.lifecycle.start()
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.lifecycle.stop()

    async def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def process(self, user_input: UserInput) -> BrainResponse | None:
        """Process a user input through the full LLM + tool-execution pipeline."""
        if self.state != AssistantState.IDLE:
            logger.warning(f"Cannot process — assistant is {self.state.name}")
            return None

        # --- INTERCEPT SYSTEM COMMANDS ---
        import re
        clean_text = re.sub(r'[^a-z\s]', '', user_input.text.lower()).strip()
        words = clean_text.split()
        
        is_standby = getattr(self, 'is_standby', False)
        if is_standby:
            if not any(w in clean_text for w in ["prism", "wake up"]):
                return None
            else:
                self.is_standby = False
        else:
            if clean_text in ["sleep", "stand by"]:
                self.is_standby = True
                
        if any(cmd in clean_text for cmd in ["follow my mouse", "stop following my mouse", "stop following"]) or "fmm" in words or "sf" in words:
            logger.info(f"System command intercepted: {user_input.text}")
            return None
        # ---------------------------------

        try:
            await self.event_bus.publish(InputReceived(input_text=user_input.text, source=user_input.source))
            await self.set_state(AssistantState.PROCESSING, reason="input received")

            self.context_manager.add_user_message(user_input.text)

            iteration = 0
            final_response: BrainResponse | None = None

            while iteration < MAX_TOOL_ITERATIONS:
                iteration += 1

                await self.set_state(AssistantState.THINKING, reason="calling LLM")
                if iteration == 1:
                    await self.event_bus.publish(ProcessingStarted())
                    await self.event_bus.publish(AssistantThinkingStarted())

                messages = self.context_manager.get_messages()
                tools = self._build_tool_schemas()

                if clean_text in ["sleep", "stand by"] and iteration == 1:
                    response = BrainResponse(text="Standing by.", tool_calls=[])
                elif any(w in clean_text for w in ["prism", "wake up"]) and iteration == 1:
                    response = BrainResponse(text="I am back online sir.", tool_calls=[])
                elif (
                    re.search(r"\b(?:use|switch\s+to)\s+api(?:\s+models?)?\b|\buse\s+cloud(?:\s+models?)?\b", clean_text)
                    and iteration == 1
                ):
                    if hasattr(self.llm_provider, "set_mode"):
                        self.llm_provider.set_mode("api")
                    desc = (
                        self.llm_provider.get_status_description()
                        if hasattr(self.llm_provider, "get_status_description")
                        else "API models"
                    )
                    response = BrainResponse(text=f"Switched to {desc}.", tool_calls=[])
                elif (
                    re.search(r"\b(?:use|switch\s+to)\s+local(?:\s+models?)?\b|\buse\s+ollama\b|\buse\s+offline(?:\s+models?)?\b", clean_text)
                    and iteration == 1
                ):
                    if hasattr(self.llm_provider, "set_mode"):
                        self.llm_provider.set_mode("local")
                    desc = (
                        self.llm_provider.get_status_description()
                        if hasattr(self.llm_provider, "get_status_description")
                        else "local models"
                    )
                    response = BrainResponse(text=f"Switched to {desc}.", tool_calls=[])
                elif (
                    clean_text in ["model status", "which model are you using", "what model are you using", "active model"]
                    and iteration == 1
                ):
                    desc = (
                        self.llm_provider.get_status_description()
                        if hasattr(self.llm_provider, "get_status_description")
                        else "Standard LLM Provider"
                    )
                    response = BrainResponse(text=f"Currently running: {desc}.", tool_calls=[])
                else:
                    response = await self.llm_provider.generate_response(
                        messages, tools or None
                    )

                self.context_manager.add_assistant_message(
                    text=response.text, tool_calls=response.tool_calls
                )

                if response.tool_calls:
                    await self.set_state(AssistantState.EXECUTING_TOOL, reason="running tools")
                    results = await asyncio.gather(
                        *[self.tool_executor.execute(tc) for tc in response.tool_calls]
                    )
                    for result in results:
                        self.context_manager.add_tool_result(result)
                else:
                    final_response = response
                    break

            if iteration >= MAX_TOOL_ITERATIONS:
                logger.warning("Max tool iterations reached")

            if final_response and final_response.text:
                await self.event_bus.publish(AssistantThinkingCompleted())
                await self.set_state(AssistantState.RESPONDING, reason="final response ready")
                await self.event_bus.publish(ResponseGenerated(response_text=final_response.text))

            await self.event_bus.publish(ProcessingCompleted())
            await self.set_state(AssistantState.IDLE, reason="processing complete")
            return final_response

        except asyncio.CancelledError:
            await self.set_state(AssistantState.IDLE, reason="cancelled")
            raise
        except Exception as e:
            logger.exception("Error in processing pipeline")
            await self.event_bus.publish(ErrorOccurred(error_message=str(e), exception=e))
            await self.set_state(AssistantState.ERROR, reason=str(e))
            await self.set_state(AssistantState.IDLE, reason="recovered")
            return None

    def _build_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["parameters"],
                },
            }
            for s in self.tool_registry.get_all_schemas()
        ]
