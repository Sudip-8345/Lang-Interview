import os
import asyncio

from utils.logger import get_logger
from utils.audio import save_to_temp_wav, cleanup_temp_file

logger = get_logger(__name__)

# Global cached model
_whisper_model = None

# Cache directory for Whisper model
WHISPER_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "whisper")

# Confidence threshold - below this, ask user to repeat
CONFIDENCE_THRESHOLD = 0.4


async def _load_whisper(model_name: str = "base"):
    global _whisper_model
    
    if _whisper_model is None:
        os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
        logger.info(f"Loading Whisper model: {model_name} (cache: {WHISPER_CACHE_DIR})")
        
        import whisper
        _whisper_model = await asyncio.to_thread(
            whisper.load_model,
            model_name,
            download_root=WHISPER_CACHE_DIR
        )
        logger.info("Whisper model loaded successfully")
    
    return _whisper_model


async def transcribe_with_whisper(audio_path: str, model_name: str = "base") -> tuple:
    model = await _load_whisper(model_name)
    
    result = await asyncio.to_thread(
        model.transcribe,
        audio_path,
        fp16=False,
        language="en"
    )
    
    text = result.get("text", "").strip()
    
    # Calculate confidence from segments
    segments = result.get("segments", [])
    if segments:
        avg_logprob = sum(s.get("avg_logprob", -1) for s in segments) / len(segments)
        # Convert log probability to confidence (0-1 scale)
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
    else:
        confidence = 0.5
    
    return text, confidence


async def transcribe_with_google(audio_path: str) -> str:
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    
    def recognize():
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    
    text = await asyncio.to_thread(recognize)
    return text.strip()


async def transcribe(audio_data: bytes, format_hint: str = None) -> str:

    temp_path = None
    
    try:
        temp_path = await save_to_temp_wav(audio_data, format_hint)
        
        # Try Whisper first
        try:
            text, confidence = await transcribe_with_whisper(temp_path)
            logger.info(f"Whisper: '{text[:50]}...' (confidence: {confidence:.2f})")
            
            # If confidence is too low, signal to ask user to repeat
            if confidence < CONFIDENCE_THRESHOLD and len(text) > 0:
                logger.warning(f"Low confidence ({confidence:.2f}), asking to repeat")
                return "[LOW_CONFIDENCE]"
            
            return text
            
        except Exception as e:
            logger.warning(f"Whisper failed: {e}, trying Google fallback...")
        
        # Fallback to Google
        text = await transcribe_with_google(temp_path)
        logger.info(f"Google STT: '{text[:50]}...'")
        return text
        
    except Exception as e:
        logger.error(f"All STT engines failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
        
    finally:
        cleanup_temp_file(temp_path)


# Convenience alias for backward compatibility
async def speech_to_text_async(audio_data: bytes) -> str:
    return await transcribe(audio_data)


def preload_whisper_model(model_name: str = "base"):
    import threading
    
    def _preload():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_load_whisper(model_name))
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to preload whisper model: {e}")
    
    thread = threading.Thread(target=_preload, daemon=True)
    thread.start()
    logger.info(f"Preloading Whisper model '{model_name}' in background...")
