import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

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

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

# Autonomous driving state (protected by data_lock)
auto_state = {
    'tap_end_time': 0.0,
    'trailing_car_detected': False,
}

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                # You may disable this if you don't need to display the frames / This could effect the fps
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def _contour_cx(contour):
    M = cv2.moments(contour)
    return int(M['m10'] / M['m00']) if M['m00'] > 0 else None


def _detect_tokens(frame, h, start_frac=0.45, end_frac=0.90):
    """
    Scan a window from start_frac to end_frac of frame height.
    In the perspective view, bottom = close, top = far.
    0.45–0.90 covers roughly one car length ahead, skipping distant horizon tokens.
    Returns (valid_green, valid_danger) where danger = red + yellow.
    """
    roi = cv2.cvtColor(frame[int(h * start_frac):int(h * end_frac), :], cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    # Green: tighter hue range, high saturation to avoid grass/road noise
    green_mask = cv2.morphologyEx(
        cv2.inRange(roi, np.array([42, 80, 80]), np.array([78, 255, 255])),
        cv2.MORPH_OPEN, kernel
    )
    # Red: two hue segments (wraps around 0/180), high saturation for accuracy
    red_mask = cv2.morphologyEx(
        cv2.bitwise_or(
            cv2.inRange(roi, np.array([0,   100, 100]), np.array([8,   255, 255])),
            cv2.inRange(roi, np.array([172, 100, 100]), np.array([180, 255, 255]))
        ),
        cv2.MORPH_OPEN, kernel
    )
    # Yellow: narrow hue band, high saturation to avoid headlights/road markings
    yellow_mask = cv2.morphologyEx(
        cv2.inRange(roi, np.array([22, 100, 100]), np.array([33, 255, 255])),
        cv2.MORPH_OPEN, kernel
    )
    danger_mask = cv2.bitwise_or(red_mask, yellow_mask)

    MIN_AREA = 200
    gc, _ = cv2.findContours(green_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dc, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (
        [c for c in gc if cv2.contourArea(c) > MIN_AREA],
        [c for c in dc if cv2.contourArea(c) > MIN_AREA],
    )


def _detect_road_offset(frame, h, w):
    """
    Estimate lateral drift using red road barriers visible in the lower third.
    Returns pixel offset: positive means car has drifted left (steer right to correct).
    """
    lower = cv2.cvtColor(frame[2 * h // 3:, :], cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(lower, np.array([0,   100, 100]), np.array([10,  255, 255])),
        cv2.inRange(lower, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 100]
    if len(valid) < 2:
        return 0
    xs = sorted([_contour_cx(c) for c in valid if _contour_cx(c) is not None])
    if len(xs) < 2:
        return 0
    return ((xs[0] + xs[-1]) // 2) - (w // 2)


def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        tap_end     = auto_state['tap_end_time']
        trailing    = auto_state['trailing_car_detected']

    if front_frame is None:
        return

    now = time.time()
    # Only hard-block during a trailing-car dodge — green can override everything else
    if now < tap_end and trailing:
        return

    h, w = front_frame.shape[:2]
    center_x = w // 2
    GREEN_TAP    = 0.05   # short tap: re-evaluate quickly while chasing
    DANGER_TAP   = 0.05   # fast avoidance tap
    TRAILING_TAP = 0.15   # full tap: decisive dodge away from trailing car

    valid_green, valid_danger = _detect_tokens(front_frame, h)
    road_offset  = _detect_road_offset(front_frame, h, w)
    green_target = max(valid_green, key=cv2.contourArea) if valid_green else None

    # -------------------------------------------------------------------
    # Autonomous steering — tap-based (slide 9)
    # Priority: trailing car > green token > danger (red+yellow) > lane centering
    # Green always overrides danger — never sacrifice a green for avoidance
    # -------------------------------------------------------------------
    tap_value    = 0.0
    tap_duration = 0.0

    if trailing:
        tap_value    = 1.0
        tap_duration = TRAILING_TAP

    elif green_target is not None:
        # Chase green across all lanes — full frame width detected
        gx = _contour_cx(green_target)
        if gx is not None:
            offset = gx - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP

    elif valid_danger:
        # Avoid red + yellow as fast as possible when no green is visible
        avoidance = 0.0
        total_w   = 0.0
        for c in valid_danger:
            cx_val = _contour_cx(c)
            if cx_val is None:
                continue
            area   = cv2.contourArea(c)
            offset = cx_val - center_x
            avoidance += (-offset / center_x) * area
            total_w   += area
        if total_w > 0:
            raw = avoidance / total_w
            tap_value = float(np.clip(raw * 2.5, -1.0, 1.0))
            if 0 < abs(tap_value) < 0.5:
                tap_value = 0.5 * np.sign(tap_value)
            tap_duration = DANGER_TAP

    elif abs(road_offset) > w * 0.08:
        tap_value    = float(np.clip(road_offset / (w * 0.3), -0.5, 0.5))
        tap_duration = GREEN_TAP

    # --- Debug overlay ---
    debug = front_frame.copy()
    for c in valid_green:
        cv2.drawContours(debug, [c], -1, (0, 255, 0), 2)
    for c in valid_danger:
        cv2.drawContours(debug, [c], -1, (0, 0, 255), 2)
    cv2.putText(debug,
                f"Tap:{tap_value:+.2f}  G:{len(valid_green)} D:{len(valid_danger)}  Trail:{trailing}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Token Detection", cv2.resize(debug, (640, 480)))
    cv2.waitKey(1)

    with data_lock:
        shared_data['acceleration_input'] = 1.0
        if abs(tap_value) > 0.05:
            shared_data['steering_input'] = float(tap_value)
            auto_state['tap_end_time']    = now + tap_duration
        else:
            shared_data['steering_input'] = 0.0


def back_camera_processing_task():
    # Detect trailing cars approaching from behind (Trailing Car event)
    with data_lock:
        back_frame = shared_data['latest_back_frame']

    if back_frame is None:
        return

    h, w = back_frame.shape[:2]
    # Focus on center-lower region where a close trailing car appears large
    roi = back_frame[h // 2:, w // 4: 3 * w // 4]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Cars have saturated colors; exclude road gray, sky blue, and grass green
    car_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    not_blue  = cv2.bitwise_not(cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])))
    not_green = cv2.bitwise_not(cv2.inRange(hsv, np.array([35,  40, 40]), np.array([85,  255, 255])))
    car_mask  = cv2.bitwise_and(car_mask, cv2.bitwise_and(not_blue, not_green))

    car_pixels = int(np.sum(car_mask > 0))
    trailing = car_pixels > 3000

    with data_lock:
        auto_state['trailing_car_detected'] = trailing


def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    # steering_input: -1.0 (full left) to 1.0 (full right)
    # acceleration_input: -1.0 (full reverse) to 1.0 (full forward)
    now = time.time()
    with data_lock:
        # Release steering tap when it expires (return to centre)
        if now >= auto_state['tap_end_time']:
            shared_data['steering_input'] = 0.0
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera    = RTTask("ReadFrontCamera",    period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
    t_back_camera     = RTTask("ReadBackCamera",     period=0.005, priority=TaskPriority.HIGH,   execute_func=read_back_camera_task)
    t_back_processing = RTTask("BackCameraProcess",  period=0.033, priority=TaskPriority.MEDIUM, execute_func=back_camera_processing_task)
    t_processing      = RTTask("Processing",         period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls        = RTTask("SendControls",       period=0.005, priority=TaskPriority.HIGH,   execute_func=send_controls_task)

    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_back_processing.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_back_processing.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
