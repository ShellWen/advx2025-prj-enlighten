import threading
import time

import cv2
from loguru import logger

from signals import camera_frame
from utils import signal_async_wrapper

_is_inited = False
_cap = None
_camera_thread = None
_stop_event = None


def init_camera():
    logger.info("Initializing camera module...")
    global _is_inited, _cap
    if _is_inited:
        logger.warning("Camera module is already initialized.")
        return _cap
    try:
        _cap = cv2.VideoCapture(0)
        if not _cap.isOpened():
            logger.error("Could not open camera.")
            raise Exception("Camera not accessible")
        _is_inited = True
        logger.info("Camera module initialized successfully.")
        return _cap
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        raise


def _camera_worker():
    logger.info("Camera worker thread started...")
    global _cap, _stop_event
    if not _is_inited or _cap is None:
        logger.error("Camera not initialized. Call init_camera() first.")
        return

    cap: cv2.VideoCapture = _cap
    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from camera.")
                break
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow('Camera Frame', frame)
            camera_frame.send(frame, _async_wrapper=signal_async_wrapper)
            cv2.waitKey(1)
            time.sleep(0.033)  # ~30 FPS
        logger.info("Camera worker thread stopped.")
    except Exception as e:
        logger.error(f"Error running camera: {e}")
    finally:
        if _cap:
            cap.release()
        cv2.destroyAllWindows()


def start_camera():
    global _camera_thread, _stop_event
    if _camera_thread and _camera_thread.is_alive():
        logger.warning("Camera thread is already running.")
        return

    _stop_event = threading.Event()
    _camera_thread = threading.Thread(target=_camera_worker, daemon=True)
    _camera_thread.start()
    logger.info("Camera thread started.")


def stop_camera():
    global _stop_event, _camera_thread
    if _stop_event:
        _stop_event.set()
    if _camera_thread and _camera_thread.is_alive():
        _camera_thread.join(timeout=5.0)
        logger.info("Camera thread stopped.")
