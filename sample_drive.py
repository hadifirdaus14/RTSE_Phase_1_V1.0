import socket
import threading
import cv2
import time
import ctypes
import keyboard
import numpy as np

import shared as _sh

# Convenience aliases so task code can use these names directly
shared_data = _sh.shared_data
data_lock   = _sh.data_lock

# Re-export constants
CAMERA_HOST       = _sh.CAMERA_HOST
FRONT_CAMERA_PORT = _sh.FRONT_CAMERA_PORT
BACK_CAMERA_PORT  = _sh.BACK_CAMERA_PORT
CONTROL_HOST      = _sh.CONTROL_HOST
CONTROL_PORT      = _sh.CONTROL_PORT

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this)
# ---------------------------------------------------------
class TaskPriority:
    HIGH   = 1
    MEDIUM = 2
    LOW    = 3

class RTTask(threading.Thread):
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name         = name
        self.period       = period
        self.priority     = priority
        self.execute_func = execute_func
        self.daemon       = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception:
            pass

        while _sh.is_running:
            start_time = time.time()
            self.execute_func()
            exec_time  = time.time() - start_time
            sleep_time = self.period - exec_time
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this)
# ---------------------------------------------------------
def setup_cameras():
    print("Connecting to Cameras...")
    front_connected = False
    back_connected  = False

    while _sh.is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((_sh.CAMERA_HOST, _sh.FRONT_CAMERA_PORT))
                _sh.front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((_sh.CAMERA_HOST, _sh.BACK_CAMERA_PORT))
                _sh.back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((_sh.CONTROL_HOST, _sh.CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {_sh.CONTROL_HOST}:{_sh.CONTROL_PORT}")

    while _sh.is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            _sh.control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------
from camera_tasks import read_front_camera_task, read_back_camera_task
from tasks import processing_task, back_camera_processing_task, send_controls_task

# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    # Period  = execution interval in seconds
    # Priority = task scheduling priority (HIGH > MEDIUM > LOW)
    t_front_camera    = RTTask("ReadFrontCamera",   period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
    t_back_camera     = RTTask("ReadBackCamera",    period=0.005, priority=TaskPriority.HIGH,   execute_func=read_back_camera_task)
    t_back_processing = RTTask("BackCameraProcess", period=0.033, priority=TaskPriority.MEDIUM, execute_func=back_camera_processing_task)
    t_processing      = RTTask("Processing",        period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls        = RTTask("SendControls",      period=0.005, priority=TaskPriority.HIGH,   execute_func=send_controls_task)

    t_front_camera.start()
    t_back_camera.start()
    t_back_processing.start()
    t_processing.start()
    t_controls.start()

    try:
        while _sh.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        _sh.is_running = False

    t_front_camera.join()
    t_back_camera.join()
    t_back_processing.join()
    t_processing.join()
    t_controls.join()

    if _sh.front_camera_sock:
        _sh.front_camera_sock.close()
    if _sh.back_camera_sock:
        _sh.back_camera_sock.close()
    if _sh.control_conn:
        _sh.control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
