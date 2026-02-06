import io
import os
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from utils.logger import get_logger
logger = get_logger(__name__)

async def load_audio(audio_data: bytes):
    # Try soundfile first 
    try:
        with io.BytesIO(audio_data) as buf:
            data, sampling_rate = sf.read(buf)
            return data, sampling_rate
    except:
        pass    
    
    # Fallback to pydub 
    with io.BytesIO(audio_data) as buf:
        audio = AudioSegment.from_file(buf)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (audio.sample_width * 8 - 1))
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        return samples, audio.frame_rate
    
async def save_to_temp_wav(audio_data: bytes, format_hint: str = None) -> str:
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp.name
    temp.close()
    
    samples, sampling_rate = await load_audio(audio_data)
    sf.write(temp_path, samples, sampling_rate)
    return temp_path

async def convert_to_format(audio_data: bytes, output_format: str = "mp3") -> bytes:
    with io.BytesIO(audio_data) as buf:
        audio = AudioSegment.from_file(buf)
    
    out = io.BytesIO()
    audio.export(out, format=output_format)
    return out.getvalue()

async def get_audio_duration(audio_data: bytes) -> float:
    try:
        with io.BytesIO(audio_data) as buf:
            audio = AudioSegment.from_file(buf)
            return len(audio) / 1000.0  # milliseconds to seconds
    except:
        return 0.0
    
def cleanup_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except:
        pass