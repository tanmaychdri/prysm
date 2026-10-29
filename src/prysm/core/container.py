from prysm.audio.pipeline import VoicePipeline
import logging

from prysm.brain.context import ContextManager
from prysm.brain.router import RouterLLMProvider
from prysm.config.settings import Settings
from prysm.core.assistant import PrysmAssistant
from prysm.core.events import EventBus
from prysm.memory.long_term import LongTermMemory
from prysm.memory.store import ConversationStore
from prysm.tools.executor import ToolExecutor
from prysm.tools.memory_tools import MemoryTools
from prysm.tools.os.app import OsAppTools
from prysm.tools.os.clipboard import OsClipboardTools
from prysm.tools.os.power import OsPowerTools
from prysm.tools.os.system import OsSystemTools
from prysm.tools.os.volume import OsVolumeTools
from prysm.tools.registry import ToolRegistry
from prysm.tools.user_tools import UserManagementTools
from prysm.users.session import UserSession
from prysm.users.store import UserStore

logger = logging.getLogger(__name__)


class ApplicationContainer:
    """
    Dependency injection container.
    Wires all concrete implementations together.

    Audio components are only built when build_voice() is called,
    so chat mode starts instantly without loading STT/TTS.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

        # Core infrastructure
        self.event_bus = EventBus()
        self.tool_registry = ToolRegistry()

        # Users
        self.user_store = UserStore()
        self.user_session = UserSession()

        # Register OS tools
        OsSystemTools().register(self.tool_registry)
        OsAppTools().register(self.tool_registry)
        OsPowerTools().register(self.tool_registry)
        OsVolumeTools().register(self.tool_registry)
        OsClipboardTools().register(self.tool_registry)

        # Memory
        self.long_term_memory = LongTermMemory()
        self.conversation_store = ConversationStore()
        MemoryTools(self.long_term_memory, self.user_session).register(self.tool_registry)

        # User management tools
        UserManagementTools(self.user_store, self.user_session).register(self.tool_registry)

        # Brain
        self.context_manager = ContextManager(
            long_term_memory=self.long_term_memory,
            conversation_store=self.conversation_store,
            session=self.user_session,
        )
        self.tool_executor = ToolExecutor(self.tool_registry, session=self.user_session)
        self.llm_provider = RouterLLMProvider(self.settings)

        # Register model routing tools
        from prysm.tools.model_tools import ModelRoutingTools
        ModelRoutingTools(self.llm_provider).register(self.tool_registry)

        # Assistant
        self.assistant = PrysmAssistant(
            event_bus=self.event_bus,
            tool_registry=self.tool_registry,
            llm_provider=self.llm_provider,
            context_manager=self.context_manager,
            tool_executor=self.tool_executor,
        )

        # Voice pipeline — not built until build_voice() is called
        self.voice_pipeline = None

        logger.info(
            f"Container ready | {self.llm_provider.get_status_description()} | "
            f"Tools: {len(self.tool_registry.list_tools())}"
        )

    def build_voice(self) -> "VoicePipeline":  # noqa: F821
        """
        Lazily build all audio components and the voice pipeline.
        Only called in voice mode — keeps chat mode fast and silent.
        """
        from prysm.audio.capture import SoundDeviceCapture, SoundDeviceOutput
        from prysm.audio.pipeline import VoicePipeline
        from prysm.audio.providers.elevenlabs_tts import ElevenLabsTTS
        from prysm.audio.providers.faster_whisper_stt import FasterWhisperSTT
        from prysm.audio.vad import EnergyVAD
        from prysm.audio.wakeword import EnergyWakeWordDetector

        self.audio_in = SoundDeviceCapture(self.settings.audio)
        self.audio_out = SoundDeviceOutput(self.settings.audio)
        self.wake_word = EnergyWakeWordDetector(self.settings.wakeword)
        self.vad = EnergyVAD(self.settings.vad, self.settings.audio.sample_rate)
        self.stt = FasterWhisperSTT(self.settings.stt)
        self.tts = ElevenLabsTTS(self.settings)

        self.voice_pipeline = VoicePipeline(
            settings=self.settings,
            audio_in=self.audio_in,
            audio_out=self.audio_out,
            wake_word=self.wake_word,
            vad=self.vad,
            stt=self.stt,
            tts=self.tts,
            event_bus=self.event_bus,
            assistant=self.assistant,
        )

        logger.info(f"Voice pipeline ready | STT: {self.settings.stt.model}")
        return self.voice_pipeline
