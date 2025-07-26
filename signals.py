import asyncio
import threading
from typing import Any

from blinker import signal
from cv2 import Mat
from numpy import ndarray, dtype

FrameType = Mat | ndarray[Any, dtype]

camera_frame = signal("camera_frame")
speech_recognition_result = signal("speech_recognition_result")
keyword_detected = signal("keyword_detected")


def send_async_from_thread(signal_obj, data=None):
    """从其他线程安全地发送异步信号"""
    def _send():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(signal_obj.send_async(data))
        finally:
            loop.close()
    
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
