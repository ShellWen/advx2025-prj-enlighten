import asyncio
import re
import signal
import sys

from loguru import logger

import camera
import llm
import speech_recognition
import utils
from signals import speech_recognition_result, keyword_detected, camera_frame, FrameType
import sounddevice as sd

_last_frame: FrameType | None = None


async def camera_frame_handler(sender: FrameType):
    global _last_frame
    
    _last_frame = sender


async def speech_recognition_handler(sender: str):
    logger.info(sender)
    msg = re.sub(r'\s+', '', sender)
    if "看到" in msg or "看见" in msg:
        logger.info("Detected keyword in speech recognition result.")
        await keyword_detected.send_async()


async def keyword_detected_handler(sender: None):
    if _last_frame is None:
        logger.warning("No frame available to process.")
        return
    logger.info("Keyword detected, processing last frame.")
    tts_mp3 = await llm.tts("请让我看看")
    tts_pcm, samplerate = utils.mp3_to_numpy_array(tts_mp3)
    sd.play(tts_pcm, samplerate)
    image_caption = await llm.ask_image_caption(_last_frame)
    logger.info(f"Image caption: {image_caption}")
    tts_mp3 = await llm.tts(image_caption)
    tts_pcm, samplerate = utils.mp3_to_numpy_array(tts_mp3)
    sd.play(tts_pcm, samplerate)


def signal_handler(sig, frame):
    logger.info("Received interrupt signal, stopping threads...")
    camera.stop_camera()
    speech_recognition.stop_speech_recognition()
    sys.exit(0)


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    camera_frame.connect(camera_frame_handler)
    speech_recognition_result.connect(speech_recognition_handler)
    keyword_detected.connect(keyword_detected_handler)

    camera.init_camera()
    speech_recognition.init_speech_recognition()
    
    camera.start_camera()
    speech_recognition.start_speech_recognition(device=1)
    
    logger.info("All threads started, main loop running...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping...")
    finally:
        camera.stop_camera()
        speech_recognition.stop_speech_recognition()


if __name__ == "__main__":
    asyncio.run(main())
