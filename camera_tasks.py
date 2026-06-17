import socket
import time
import select
import cv2
import numpy as np

import sample_drive as _sd
from state import reconnect_state


def read_single_camera(sock, window_name, data_key):
    """Read the latest frame from a camera socket and store it in shared_data."""
    if sock is None:
        return

    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length   = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and _sd.is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet

        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes

        while _sd.is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break

            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length   = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and _sd.is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes

        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with _sd.data_lock:
                    _sd.shared_data[data_key] = frame
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)

    except Exception:
        pass


def read_front_camera_task():
    if _sd.front_camera_sock is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((_sd.CAMERA_HOST, _sd.FRONT_CAMERA_PORT))
            _sd.front_camera_sock = s
            print("[Camera] Front camera reconnected.")
        except Exception:
            pass
        return

    with _sd.data_lock:
        prev = _sd.shared_data['latest_front_frame']
    read_single_camera(_sd.front_camera_sock, "Front Camera", 'latest_front_frame')
    with _sd.data_lock:
        curr = _sd.shared_data['latest_front_frame']

    if curr is prev:
        if reconnect_state['front_stale_since'] == 0.0:
            reconnect_state['front_stale_since'] = time.time()
        elif time.time() - reconnect_state['front_stale_since'] > 2.0:
            try:
                _sd.front_camera_sock.close()
            except Exception:
                pass
            _sd.front_camera_sock = None
            reconnect_state['front_stale_since'] = 0.0
            print("[Camera] Front camera disconnected.")
    else:
        reconnect_state['front_stale_since'] = 0.0


def read_back_camera_task():
    if _sd.back_camera_sock is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((_sd.CAMERA_HOST, _sd.BACK_CAMERA_PORT))
            _sd.back_camera_sock = s
            print("[Camera] Back camera reconnected.")
        except Exception:
            pass
        return

    with _sd.data_lock:
        prev = _sd.shared_data['latest_back_frame']
    read_single_camera(_sd.back_camera_sock, "Back Camera", 'latest_back_frame')
    with _sd.data_lock:
        curr = _sd.shared_data['latest_back_frame']

    if curr is prev:
        if reconnect_state['back_stale_since'] == 0.0:
            reconnect_state['back_stale_since'] = time.time()
        elif time.time() - reconnect_state['back_stale_since'] > 2.0:
            try:
                _sd.back_camera_sock.close()
            except Exception:
                pass
            _sd.back_camera_sock = None
            reconnect_state['back_stale_since'] = 0.0
            print("[Camera] Back camera disconnected.")
    else:
        reconnect_state['back_stale_since'] = 0.0
