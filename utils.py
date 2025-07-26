import io

import numpy as np
from loguru import logger
from pydub import AudioSegment
import asyncio


def mp3_to_numpy_array(mp3: bytes) -> (np.ndarray | None, int | None):
    try:
        fake_file = io.BytesIO(mp3)
        audio = AudioSegment.from_mp3(fake_file)
        audio = audio.set_frame_rate(44100)  # 设置采样率为 44100 Hz

        sample_rate = audio.frame_rate
        channels = audio.channels

        samples = audio.get_array_of_samples()

        if audio.sample_width == 2:  # int16 对应的 sample_width 是 2 字节
            audio_array = np.array(samples, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / (2 ** 15)
        elif audio.sample_width == 4:  # int32 对应的 sample_width 是 4 字节
            audio_array = np.array(samples, dtype=np.int32)
            audio_array = audio_array.astype(np.float32) / (2 ** 31)

        # 如果是多声道，需要重塑数组形状 (样本数, 声道数)
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels)

        return audio_array, sample_rate

    except Exception as e:
        logger.error("Error converting MP3 to numpy array.", e)
        return None


def signal_async_wrapper(func):
    def inner(*args, **kwargs):
        asyncio.run(func(*args, **kwargs))

    return inner
