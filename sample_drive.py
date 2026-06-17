import socket
import threading
import struct
import cv2
import numpy as np
import time
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
# Real-Time Scheduling Framework
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
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
# Network Connection Setup 
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
# Task Implementations 
# ---------------------------------------------------------

auto_state = {
    'tap_end_time': 0.0,
    'trailing_car_detected': False,
    'police_detected': False 
}

def read_single_camera(sock, window_name, data_key):
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
    roi_bgr = frame[int(h * start_frac):int(h * end_frac), :]
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    # Sky brightness scanner
    sky_bgr = frame[0:int(h * 0.25), :]
    sky_hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)
    is_dark = np.mean(sky_hsv[:, :, 2]) < 55  # Fine-tuned threshold

    # Ultra Night Vision masks
    green_mask = cv2.morphologyEx(cv2.inRange(roi, np.array([42, 50, 20]), np.array([78, 255, 255])), cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(cv2.bitwise_or(
        cv2.inRange(roi, np.array([0, 50, 20]), np.array([8, 255, 255])),
        cv2.inRange(roi, np.array([172, 50, 20]), np.array([180, 255, 255]))
    ), cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(cv2.inRange(roi, np.array([22, 50, 20]), np.array([33, 255, 255])), cv2.MORPH_OPEN, kernel)

    MIN_AREA = 200
    gc, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rc, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yc, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return (
        [c for c in gc if cv2.contourArea(c) > MIN_AREA],
        [c for c in rc if cv2.contourArea(c) > MIN_AREA],
        [c for c in yc if cv2.contourArea(c) > MIN_AREA],
        is_dark
    )

def _detect_road_offset(frame, h, w):
    lower = cv2.cvtColor(frame[2 * h // 3:, :], cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(lower, np.array([0,   40, 20]), np.array([10,  255, 255])),
        cv2.inRange(lower, np.array([170, 40, 20]), np.array([180, 255, 255]))
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
        police      = auto_state['police_detected']

    if front_frame is None:
        return

    now = time.time()
    is_tapping = now < tap_end

    h, w = front_frame.shape[:2]
    center_x = w // 2
    GREEN_TAP    = 0.05

    valid_green, valid_red, valid_yellow, is_dark = _detect_tokens(front_frame, h)
    road_offset  = _detect_road_offset(front_frame, h, w)
    
    if police:
        valid_danger = valid_yellow
    else:
        valid_danger = valid_red + valid_yellow 

    threats = []
    for c in valid_danger:
        cx_val = _contour_cx(c)
        if cx_val is not None:
            offset = cx_val - center_x
            # TIGHTENED THREAT BOX: Only dodge if it's dead ahead (18% width). 
            # Stops the car from phantom-dodging tokens in other lanes!
            if abs(offset) < w * 0.18: 
                threats.append(c)

    green_target = max(valid_green, key=cv2.contourArea) if valid_green else None

    tap_value    = 0.0
    tap_duration = 0.0
    current_speed = 1.0  

    # NIGHT MODE: Drop speed slightly to give the sensors time to react in the dark
    if is_dark:
        current_speed = 0.70

    if not is_tapping:
        # Priority 1: Survive trailing car
        if trailing:
            # AGGRESSIVE DODGE: 0.85 to snap out of the way of fast Chasing Cars
            tap_value = -0.85 if road_offset > 0 else 0.85
            if road_offset == 0: tap_value = 0.85
            tap_duration = 0.25 

        # Priority 2: Police Chase 
        elif police and len(valid_red) > 0:
            red_target = max(valid_red, key=cv2.contourArea)
            rx = _contour_cx(red_target)
            if rx is not None:
                offset = rx - center_x
                tap_value    = float(np.clip(offset / center_x, -0.7, 0.7))
                tap_duration = GREEN_TAP

        # Priority 3: Dodge Threats 
        elif len(threats) > 0:
            largest_threat = max(threats, key=cv2.contourArea)
            tx = _contour_cx(largest_threat)
            offset = tx - center_x
            
            # Smooth, rapid token dodge
            tap_value = -0.75 if offset > 0 else 0.75
            tap_duration = 0.15 

        # Priority 4: Chase Green
        elif green_target is not None:
            gx = _contour_cx(green_target)
            if gx is not None:
                offset = gx - center_x
                tap_value    = float(np.clip(offset / center_x, -0.6, 0.6))
                tap_duration = GREEN_TAP

        # Priority 5: Lane Centering 
        elif abs(road_offset) > w * 0.35:
            tap_value    = float(np.clip(road_offset / (w * 0.3), -0.5, 0.5))
            tap_duration = GREEN_TAP

        with data_lock:
            shared_data['acceleration_input'] = current_speed
            if abs(tap_value) > 0.05:
                shared_data['steering_input'] = float(tap_value)
                auto_state['tap_end_time']    = now + tap_duration
            else:
                shared_data['steering_input'] = 0.0
                
    else:
        with data_lock:
            shared_data['acceleration_input'] = current_speed

    # --- Debug overlay ---
    debug = front_frame.copy()
    for c in valid_green:
        cv2.drawContours(debug, [c], -1, (0, 255, 0), 2)
    for c in valid_danger:
        is_threat = any(np.array_equal(c, t) for t in threats)
        thickness = 4 if is_threat else 1
        cv2.drawContours(debug, [c], -1, (0, 0, 255), thickness)
    
    cv2.putText(debug,
                f"Tap:{tap_value:+.2f} Locked:{is_tapping} Dark:{is_dark}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Token Detection", cv2.resize(debug, (640, 480)))
    cv2.waitKey(1)

def back_camera_processing_task():
    with data_lock:
        back_frame = shared_data['latest_back_frame']

    if back_frame is None:
        return

    h, w = back_frame.shape[:2]
    roi = back_frame[h // 2:, w // 4: 3 * w // 4]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    car_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    not_blue  = cv2.bitwise_not(cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])))
    not_green = cv2.bitwise_not(cv2.inRange(hsv, np.array([35,  40, 40]), np.array([85,  255, 255])))
    car_mask  = cv2.bitwise_and(car_mask, cv2.bitwise_and(not_blue, not_green))
    
    # HYPER-SENSITIVE RADAR: Trigger instantly (100 pixels) for fast Chasing Cars
    car_pixels = int(np.sum(car_mask > 0))
    trailing = car_pixels > 100

    blue_light_mask = cv2.inRange(hsv, np.array([100, 150, 150]), np.array([130, 255, 255]))
    red_light_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
    )
    police_pixels = cv2.countNonZero(blue_light_mask) + cv2.countNonZero(red_light_mask)
    police = police_pixels > 300 

    with data_lock:
        auto_state['trailing_car_detected'] = trailing
        auto_state['police_detected'] = police

    debug_back = back_frame.copy()
    cv2.rectangle(debug_back, (w // 4, h // 2), (3 * w // 4, h), (255, 0, 0), 2)
    cv2.putText(debug_back, f"Trail: {trailing} | Police: {police}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not police else (0, 0, 255), 2)
    cv2.imshow("Back Camera Radar", cv2.resize(debug_back, (640, 480)))
    cv2.waitKey(1)

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    now = time.time()
    with data_lock:
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
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    t_front_camera.join()
    t_back_camera.join()
    t_back_processing.join()
    t_processing.join()
    t_controls.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")