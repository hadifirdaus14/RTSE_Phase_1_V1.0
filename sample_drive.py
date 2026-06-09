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

def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
    _, w = front_frame.shape[:2]

    # --- Colour masks (HSV) ---
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]),  np.array([80, 255, 255]))
    red_mask   = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    )
    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255]))
    danger_mask = cv2.bitwise_or(red_mask, yellow_mask)

    # Morphological cleanup to remove noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask  = cv2.morphologyEx(green_mask,  cv2.MORPH_OPEN, kernel)
    danger_mask = cv2.morphologyEx(danger_mask, cv2.MORPH_OPEN, kernel)

    # --- Contour detection ---
    green_contours,  _ = cv2.findContours(green_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    danger_contours, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA  = 200
    center_x  = w // 2
    steering  = 0.0
    acceleration = 1.0

    # Pick the largest green token as the collection target
    valid_green  = [c for c in green_contours  if cv2.contourArea(c) > MIN_AREA]
    valid_danger = [c for c in danger_contours if cv2.contourArea(c) > MIN_AREA]

    green_target  = max(valid_green,  key=cv2.contourArea) if valid_green  else None
    danger_target = max(valid_danger, key=cv2.contourArea) if valid_danger else None

    # Steer towards the green token
    if green_target is not None:
        M = cv2.moments(green_target)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            steering = np.clip((cx - center_x) / center_x, -1.0, 1.0)

    # Override: steer away from the nearest danger token (red/yellow)
    if danger_target is not None:
        M = cv2.moments(danger_target)
        if M['m00'] > 0:
            dx = int(M['m10'] / M['m00'])
            danger_area = cv2.contourArea(danger_target)
            if danger_area > 500:
                avoid = np.clip(-((dx - center_x) / center_x), -1.0, 1.0)
                weight = min(danger_area / 5000.0, 1.0)  # stronger avoidance as token gets larger/closer
                steering = (1.0 - weight) * steering + weight * avoid

    # --- Debug overlay ---
    debug = front_frame.copy()
    for c in valid_green:
        cv2.drawContours(debug, [c], -1, (0, 255, 0), 2)
    for c in valid_danger:
        # distinguish red vs yellow in the overlay
        mask_pt = tuple(c[0][0])
        _, g, r = front_frame[mask_pt[1], mask_pt[0]]
        colour = (0, 0, 255) if r > g else (0, 165, 255)
        cv2.drawContours(debug, [c], -1, colour, 2)
    cv2.putText(debug, f"Steer: {steering:+.2f}  Accel: {acceleration:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Token Detection", cv2.resize(debug, (640, 480)))
    cv2.waitKey(1)

    with data_lock:
        shared_data['steering_input']     = float(steering)
        shared_data['acceleration_input'] = float(acceleration)


def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    # steering_input: -1.0 (full left) to 1.0 (full right)
    # acceleration_input: -1.0 (full reverse) to 1.0 (full forward)
    with data_lock:
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
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
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
