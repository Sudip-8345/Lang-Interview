import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseModel):
    # PostgreSQL Database
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "Interview_AI")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "020304")
    
    # LLM - OpenRouter (primary)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "upstage/solar-pro-3:free")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Groq (fallback)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Google (fallback)
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # TTS settings (Edge TTS with gTTS fallback)
    tts_voice: str = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
    tts_rate: str = os.getenv("TTS_RATE", "+0%")
    tts_volume: str = os.getenv("TTS_VOLUME", "+0%")
    
    # STT settings (Whisper with Google fallback)
    stt_whisper_model: str = os.getenv("STT_WHISPER_MODEL", "base")
    stt_language: str = os.getenv("STT_LANGUAGE", "en")
    
    # Audio settings
    max_audio_duration_seconds: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "90"))
    
    # Interview defaults
    default_mode: str = os.getenv("DEFAULT_MODE", "friendly")
    default_num_questions: int = int(os.getenv("DEFAULT_NUM_QUESTIONS", "3"))
    default_num_followup: int = int(os.getenv("DEFAULT_NUM_FOLLOWUP", "2"))
    
    # Upload settings
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    
    
settings = Settings()