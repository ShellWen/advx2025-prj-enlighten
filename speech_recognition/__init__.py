import json
import queue
import threading
import time

import sounddevice as sd
from loguru import logger
from vosk import Model, KaldiRecognizer

from signals import speech_recognition_result
from utils import signal_async_wrapper

_model = Model(model_name="vosk-model-small-cn-0.22")

_is_inited = False
_speech_thread = None
_stop_event = None


def init_speech_recognition():
    logger.info("Starting speech recognition module...")
    global _is_inited
    if _is_inited:
        logger.warning("Speech recognition module is already initialized.")
        return
    try:
        _is_inited = True
        logger.info("Speech recognition module initialized successfully.")
    except Exception as e:
        logger.info(f"Error initializing speech recognition: {e}")


def _speech_worker(device=None, samplerate=None):
    logger.info("Speech recognition worker thread started...")
    global _is_inited, _stop_event
    if not _is_inited:
        logger.error("Speech recognition not initialized. Call init_speech_recognition() first.")
        return

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if not _stop_event.is_set():
            q.put(bytes(indata))

    if samplerate is None:
        device_info = sd.query_devices(device, "input")
        samplerate = int(device_info["default_samplerate"])

    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                               dtype="int16", channels=1, callback=callback):
            rec = KaldiRecognizer(_model, samplerate)
            while not _stop_event.is_set():
                try:
                    data = q.get(timeout=0.1)
                    if rec.AcceptWaveform(data):
                        result_text = json.loads(rec.Result())["text"]
                        if result_text.strip():
                            speech_recognition_result.send(result_text, _async_wrapper=signal_async_wrapper)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}")
        logger.info("Speech recognition worker thread stopped.")
    except Exception as e:
        logger.error(f"Error in speech recognition worker: {e}")


def start_speech_recognition(device=None, samplerate=None):
    global _speech_thread, _stop_event
    if _speech_thread and _speech_thread.is_alive():
        logger.warning("Speech recognition thread is already running.")
        return
    
    _stop_event = threading.Event()
    _speech_thread = threading.Thread(
        target=_speech_worker, 
        args=(device, samplerate),
        daemon=True
    )
    _speech_thread.start()
    logger.info("Speech recognition thread started.")


def stop_speech_recognition():
    global _stop_event, _speech_thread
    if _stop_event:
        _stop_event.set()
    if _speech_thread and _speech_thread.is_alive():
        _speech_thread.join(timeout=5.0)
        logger.info("Speech recognition thread stopped.")
