from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_", env_file=".env", extra="ignore")

    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    chunk_ms: int = 30

    @field_validator("input_device", "output_device", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v

    @property
    def chunk_frames(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_", env_file=".env", extra="ignore")

    model: str = "base"
    device: str = "cpu"
    language: str = "en"


class WakeWordSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WAKEWORD_", env_file=".env", extra="ignore")

    phrase: str = "hey prysm"
    # Energy threshold for simple energy-based detection
    energy_threshold: float = 0.02


class VADSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VAD_", env_file=".env", extra="ignore")

    energy_threshold: float = 0.01
    silence_duration_ms: int = 800
    min_speech_ms: int = 300


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (API Harness - Secondary)
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")

    # Local Models (Ollama) & Routing
    llm_mode: str = Field(default="local", alias="LLM_MODE")  # "local" or "api"
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    local_primary_model: str = Field(default="qwen2.5-coder:14b", alias="LOCAL_PRIMARY_MODEL")
    local_thinking_model: str = Field(default="qwen3.5:9b", alias="LOCAL_THINKING_MODEL")
    ollama_models_path: str = Field(default=r"X:\OllamaModels", alias="OLLAMA_MODELS")

    # ElevenLabs
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")

    # Nested settings
    audio: AudioSettings = Field(default_factory=AudioSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    wakeword: WakeWordSettings = Field(default_factory=WakeWordSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
