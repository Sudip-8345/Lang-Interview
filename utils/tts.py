import io
import os
import asyncio

import edge_tts
from gtts import gTTS

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _clean_text(text: str) -> str:
    replacements = {
        "**": "", "__": "", "```": "", "`": "",
        "#": "", "*": "", "\n\n": ". ", "\n": " ",
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Remove multiple spaces
    while "  " in result:
        result = result.replace("  ", " ")
    
    return result.strip()


async def synthesize_edge(text: str, voice: str = None, rate: str = None) -> bytes:
    voice = voice or settings.tts_voice
    rate = rate or settings.tts_rate
    
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate
    )
    
    audio_data = io.BytesIO()  # in-memory buffer
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])
    
    audio_data.seek(0)
    return audio_data.read()


async def synthesize_gtts(text: str, lang: str = "en") -> bytes:
    def _generate():
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        return audio_data.read()
    
    return await asyncio.to_thread(_generate)


async def synthesize(text: str, voice: str = None) -> bytes:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    clean_text = _clean_text(text)
    
    # Try Edge TTS first
    try:
        logger.info(f"Synthesizing with Edge TTS (voice={voice or settings.tts_voice})")
        audio_bytes = await synthesize_edge(clean_text, voice)
        logger.info(f"Edge TTS success: {len(audio_bytes)} bytes")
        return audio_bytes
    except Exception as e:
        logger.warning(f"Edge TTS failed: {e}, trying gTTS fallback...")
    
    # Fallback to gTTS
    try:
        logger.info("Synthesizing with gTTS")
        audio_bytes = await synthesize_gtts(clean_text)
        logger.info(f"gTTS success: {len(audio_bytes)} bytes")
        return audio_bytes
    except Exception as e:
        logger.error(f"All TTS engines failed: {e}")
        raise RuntimeError(f"TTS failed: {e}")


async def save_audio(text: str, output_path: str, voice: str = None) -> str:
    audio_bytes = await synthesize(text, voice)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    
    logger.info(f"Audio saved: {output_path}")
    return output_path


async def list_voices() -> list:
    return await edge_tts.list_voices()


async def list_english_voices() -> list:
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].startswith("en-")]


# Convenience aliases for backward compatibility
async def text_to_speech(text: str) -> bytes:
    return await synthesize(text)


def get_tts_engine():
    class SimpleTTS:
        async def synthesize(self, text: str) -> bytes:
            return await synthesize(text)
        
        async def save_to_file(self, text: str, path: str) -> str:
            return await save_audio(text, path)
    
    return SimpleTTS()
