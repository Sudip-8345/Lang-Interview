import os
import asyncio

from deepgram import DeepgramClient, AsyncDeepgramClient

from utils.logger import get_logger
from utils.audio import save_to_temp_wav, cleanup_temp_file
from utils.config import settings

logger = get_logger(__name__)

# Deepgram clients (initialized lazily)
_deepgram_client = None
_async_deepgram_client = None

# Confidence threshold - below this, ask user to repeat
CONFIDENCE_THRESHOLD = 0.4


def _get_deepgram_client():
    global _deepgram_client
    
    if _deepgram_client is None:
        api_key = settings.deepgram_api_key
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not configured")
        _deepgram_client = DeepgramClient(api_key=api_key)
        logger.info("Deepgram client initialized")
    
    return _deepgram_client


def _get_async_deepgram_client():
    global _async_deepgram_client
    
    if _async_deepgram_client is None:
        api_key = settings.deepgram_api_key
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not configured")
        _async_deepgram_client = AsyncDeepgramClient(api_key=api_key)
        logger.info("Async Deepgram client initialized")
    
    return _async_deepgram_client


async def transcribe_with_deepgram(audio_data: bytes) -> tuple:
    client = _get_async_deepgram_client()
    
    from deepgram import PrerecordedOptions, FileSource
    
    payload: FileSource = {
        "buffer": audio_data,
    }
    
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=False,
        punctuate=True,
        language="en",
    )
    
    response = await client.listen.rest.v("1").transcribe_file(payload, options)
    
    # Extract transcript and confidence
    result = response.results.channels[0].alternatives[0]
    text = result.transcript.strip()
    confidence = result.confidence if hasattr(result, 'confidence') else 0.9
    
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
    
    if not audio_data or len(audio_data) == 0:
        logger.warning("Empty audio data received")
        return ""
    
    logger.info(f"Transcribing audio: {len(audio_data)} bytes")
    
    # Try Deepgram first (faster, more reliable on cloud)
    try:
        text, confidence = await transcribe_with_deepgram(audio_data)
        
        if text:
            logger.info(f"Deepgram: '{text[:50]}...' (confidence: {confidence:.2f})")
            
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low confidence ({confidence:.2f}), asking to repeat")
                return "[LOW_CONFIDENCE]"
            
            return text
        else:
            logger.warning("Deepgram returned empty transcript")
            
    except Exception as e:
        logger.warning(f"Deepgram transcription failed: {e}")
    
    # Fallback to Google STT
    temp_path = None
    try:
        temp_path = await save_to_temp_wav(audio_data, format_hint)
        text = await transcribe_with_google(temp_path)
        logger.info(f"Google STT: '{text[:50]}...'")
        return text
        
    except Exception as e:
        logger.error(f"Google STT also failed: {e}")
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)
    
    logger.error("All STT engines failed")
    return ""


# Convenience alias for backward compatibility
async def speech_to_text_async(audio_data: bytes) -> str:
    return await transcribe(audio_data)


def preload_deepgram():
    try:
        _get_deepgram_client()
        _get_async_deepgram_client()
        logger.info("Deepgram clients ready")
    except Exception as e:
        logger.warning(f"Failed to initialize Deepgram: {e}")
