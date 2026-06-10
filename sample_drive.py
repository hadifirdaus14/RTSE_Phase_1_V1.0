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

challenge_state = {
    # Challenge 1: Low Light (once in first 10s)
    # Fail → all tokens become Yellow (simulator effect, code avoids yellow anyway)
    'low_light_active': False,
    'low_light_triggered': False,

    # Challenge 2: Chasing Car — switch lanes before collision or -50% speed (up to twice)
    'chase_count': 0,
    'chase_active': False,
    'chase_start_time': 0.0,

    # Challenge 3: Police Car — catch next Red Token in 10s or -50% speed
    'police_active': False,
    'police_start_time': 0.0,
    'police_red_picked': False,
    'police_done': False,
    'game_over': False,

    # Yellow token effect — 5s conservative mode after hitting a yellow token
    # Possible effects: tokens invisible / corrupted camera / input-output delay / next token hidden
    'yellow_effect_active': False,
    'yellow_effect_end_time': 0.0,

    # Adaptive brightness baseline — updated via EMA during normal (bright) frames
    'brightness_baseline': 100.0,

    # Permanent speed multiplier — stacks with each -50% penalty
    'speed_multiplier': 1.0,
    'game_start_time': time.time(),
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


def _detect_tokens(frame, h, start_frac=0.30, end_frac=0.90):
    """
    Scan a window from start_frac to end_frac of frame height.
    In the perspective view, bottom = close, top = far.
    0.30–0.90 gives ~one full car length of look-ahead on both left and right lanes.
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


def _detect_police_car(frame, h, w):
    """Returns (visible, collision) for a police car detected by blue lights."""
    roi = frame[int(h * 0.30):int(h * 0.75), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([100, 120, 120]), np.array([130, 255, 255])),
        cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    visible = int(np.sum(blue > 0)) > 300

    lower_hsv = cv2.cvtColor(frame[h // 2:, :], cv2.COLOR_BGR2HSV)
    lower_blue = cv2.inRange(lower_hsv, np.array([100, 120, 120]), np.array([130, 255, 255]))
    collision = int(np.sum(lower_blue > 0)) > 6000

    return visible, collision


def _detect_red_tokens_only(frame, h, start_frac=0.30, end_frac=0.90):
    """Return red-only contours in the standard scanning window."""
    roi = cv2.cvtColor(frame[int(h * start_frac):int(h * end_frac), :], cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(
        cv2.bitwise_or(
            cv2.inRange(roi, np.array([0,   100, 100]), np.array([8,   255, 255])),
            cv2.inRange(roi, np.array([172, 100, 100]), np.array([180, 255, 255]))
        ),
        cv2.MORPH_OPEN, kernel
    )
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 200]


def _red_token_in_close_range(frame, h):
    """Returns True when a red token occupies the close-range bottom band."""
    roi = cv2.cvtColor(frame[int(h * 0.80):, :], cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(roi, np.array([0,   100, 100]), np.array([8,   255, 255])),
        cv2.inRange(roi, np.array([172, 100, 100]), np.array([180, 255, 255]))
    )
    return int(np.sum(red_mask > 0)) > 500


def _yellow_token_in_close_range(frame, h):
    """Returns True when a yellow token is about to be hit (close-range bottom band).
    Yellow tokens trigger one of five 5s disruptions: invisible tokens, corrupted
    camera, camera delay, action delay, or next-token-type hidden.
    """
    roi = cv2.cvtColor(frame[int(h * 0.80):, :], cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.morphologyEx(
        cv2.inRange(roi, np.array([22, 100, 100]), np.array([33, 255, 255])),
        cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    return int(np.sum(yellow_mask > 0)) > 500


def _get_lane_info(frame, h, w, n_lanes=3):
    """
    Detect road lane boundaries from red barriers at the lower frame region.

    In perspective view the camera is on the car, so the car always appears at
    x = w/2.  The barrier positions tell us how far the car is from each road
    edge, which lets us derive which lane it occupies.
    Scans from 0.65h so both left and right barriers remain visible even when
    the car is close to one edge.

    Returns:
        car_lane   – 0-based lane index (0 = leftmost, n_lanes-1 = rightmost)
        lane_bounds – list of (left_x, right_x) pixel ranges for every lane
    """
    lower = cv2.cvtColor(frame[int(h * 0.65):, :], cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(lower, np.array([0,   100, 100]), np.array([10,  255, 255])),
        cv2.inRange(lower, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 100]
    xs    = sorted([_contour_cx(c) for c in valid if _contour_cx(c) is not None])

    road_width_est = int(w * 0.80)
    if len(xs) >= 2:
        road_left  = xs[0]
        road_right = xs[-1]
    elif len(xs) == 1:
        # One barrier visible — estimate the missing side from typical road width.
        # When near the right barrier (xs[0] > w/2) the right barrier is close
        # and prominent; derive left from it, and vice-versa.
        if xs[0] < w // 2:          # left barrier detected, estimate right
            road_left  = xs[0]
            road_right = min(xs[0] + road_width_est, w - 1)
        else:                        # right barrier detected, estimate left
            road_right = xs[0]
            road_left  = max(xs[0] - road_width_est, 0)
    else:
        road_left  = int(w * 0.10)
        road_right = int(w * 0.90)
    road_width = max(road_right - road_left, 1)
    lane_width = road_width / n_lanes

    # car_pos: car's x-position within the road (0 = left edge)
    car_pos  = (w / 2) - road_left
    car_lane = int(np.clip(car_pos / lane_width, 0, n_lanes - 1))

    lane_bounds = [
        (int(road_left + i * lane_width), int(road_left + (i + 1) * lane_width))
        for i in range(n_lanes)
    ]
    return car_lane, lane_bounds


def _assign_lane(contour, lane_bounds):
    """Return 0-based lane index for a contour's horizontal centre.
    Clips to edge lanes if the token is outside the detected road bounds."""
    cx = _contour_cx(contour)
    if cx is None:
        return -1
    for i, (lx, rx) in enumerate(lane_bounds):
        if lx <= cx < rx:
            return i
    return 0 if cx < lane_bounds[0][0] else len(lane_bounds) - 1


def processing_task():
    with data_lock:
        front_frame   = shared_data['latest_front_frame']
        tap_end       = auto_state['tap_end_time']
        trailing      = auto_state['trailing_car_detected']
        game_over     = challenge_state['game_over']

    if front_frame is None:
        return

    now = time.time()

    # Game over: halt the car
    if game_over:
        with data_lock:
            shared_data['acceleration_input'] = -1.0
            shared_data['steering_input'] = 0.0
        return

    h, w = front_frame.shape[:2]
    center_x = w // 2
    GREEN_TAP    = 0.05
    DANGER_TAP   = 0.05
    TRAILING_TAP = 0.15

    # -----------------------------------------------------------------------
    # Challenge 1: Low Light — detect brightness drop via adaptive EMA baseline.
    # When brightness falls below 50% of the established baseline the light is
    # considered OFF; send acceleration_input = -1.0 to turn it back ON.
    # Fail to respond → simulator turns all tokens Yellow (bad random effects).
    # -----------------------------------------------------------------------
    with data_lock:
        c1_triggered = challenge_state['low_light_triggered']

    if not c1_triggered:
        gray           = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray))

        with data_lock:
            baseline = challenge_state['brightness_baseline']

            if not challenge_state['low_light_active']:
                # Keep baseline updated only during normal bright conditions
                challenge_state['brightness_baseline'] = baseline * 0.95 + avg_brightness * 0.05

                # Trigger when brightness drops to less than half the established baseline
                if avg_brightness < baseline * 0.50 and baseline > 30:
                    challenge_state['low_light_active'] = True
                    print(f"[C1] Low light! {avg_brightness:.1f} < 50% of baseline {baseline:.1f} — turning light ON.")
            else:
                # Recovery: brightness returned to at least 70% of baseline
                if avg_brightness >= baseline * 0.70:
                    challenge_state['low_light_active'] = False
                    challenge_state['low_light_triggered'] = True
                    print(f"[C1] Light ON. brightness={avg_brightness:.1f} — resuming green token collection.")

    with data_lock:
        low_light = challenge_state['low_light_active']

    if low_light:
        # Send recovery signal; send_controls_task applies the -10% multiplier
        with data_lock:
            shared_data['acceleration_input'] = -1.0
            shared_data['steering_input'] = 0.0
        return

    # -----------------------------------------------------------------------
    # Challenge 3: Police Car — detect blue lights, track 10s timer, seek red
    # -----------------------------------------------------------------------
    police_visible, police_collision = _detect_police_car(front_frame, h, w)

    with data_lock:
        p_active     = challenge_state['police_active']
        p_done       = challenge_state['police_done']
        p_red_picked = challenge_state['police_red_picked']

    if police_visible and not p_active and not p_done:
        with data_lock:
            challenge_state['police_active']     = True
            challenge_state['police_start_time'] = now
            print("[C3] Police car spotted! Collect a red token within 10s.")
        p_active = True

    if p_active:
        if police_collision:
            with data_lock:
                challenge_state['game_over'] = True
                print("[C3] GAME OVER — collided with police car!")
            with data_lock:
                shared_data['acceleration_input'] = -1.0
                shared_data['steering_input'] = 0.0
            return

        with data_lock:
            p_start = challenge_state['police_start_time']

        if (now - p_start) >= 10.0 and not p_red_picked:
            with data_lock:
                challenge_state['police_active']     = False
                challenge_state['police_done']       = True
                challenge_state['speed_multiplier'] *= 0.5
                print(f"[C3] Timer expired — no red token! Speed×0.5 → {challenge_state['speed_multiplier']:.2f}")
            p_active = False
        elif p_red_picked:
            with data_lock:
                challenge_state['police_active'] = False
                challenge_state['police_done']   = True
                print("[C3] Red token collected — police challenge cleared!")
            p_active = False
        elif _red_token_in_close_range(front_frame, h):
            with data_lock:
                challenge_state['police_red_picked'] = True
                p_red_picked = True

    # -----------------------------------------------------------------------
    # Yellow Token Effect — 5s conservative mode after hitting a yellow token.
    # Possible simulator effects: tokens invisible / corrupted camera /
    # camera delay / action delay / next token type hidden.
    # Code response: detect collection, reduce aggressiveness for 5s.
    # -----------------------------------------------------------------------
    if _yellow_token_in_close_range(front_frame, h):
        with data_lock:
            if not challenge_state['yellow_effect_active']:
                challenge_state['yellow_effect_active']  = True
                challenge_state['yellow_effect_end_time'] = now + 5.0
                print("[Yellow] Yellow token hit — 5s disruption mode active.")

    with data_lock:
        if challenge_state['yellow_effect_active'] and now > challenge_state['yellow_effect_end_time']:
            challenge_state['yellow_effect_active'] = False
            print("[Yellow] Yellow effect expired.")
        yellow_effect = challenge_state['yellow_effect_active']

    # Only hard-block during a trailing-car dodge — green can override everything else
    if now < tap_end and trailing:
        return

    valid_green, valid_danger = _detect_tokens(front_frame, h)
    red_targets = _detect_red_tokens_only(front_frame, h) if p_active else []
    road_offset = _detect_road_offset(front_frame, h, w)
    red_target  = max(red_targets, key=cv2.contourArea) if red_targets else None

    # -------------------------------------------------------------------
    # Lane-aware steering
    # Classify every token into its road lane; only react to tokens that are
    # actually in the car's own lane.  Tokens in adjacent lanes are ignored
    # unless the car needs to switch for a better outcome.
    #
    # Priority: trailing car > police red-seek > current-lane danger switch
    #           > current-lane green stay > seek nearest safe green lane
    #           > road-centre correction
    # -------------------------------------------------------------------
    car_lane, lane_bounds = _get_lane_info(front_frame, h, w)
    n_lanes = len(lane_bounds)

    lane_green  = [[] for _ in range(n_lanes)]
    lane_danger = [[] for _ in range(n_lanes)]
    for c in valid_green:
        li = _assign_lane(c, lane_bounds)
        if 0 <= li < n_lanes:
            lane_green[li].append(c)
    for c in valid_danger:
        li = _assign_lane(c, lane_bounds)
        if 0 <= li < n_lanes:
            lane_danger[li].append(c)

    current_safe  = not bool(lane_danger[car_lane])
    current_green = bool(lane_green[car_lane])

    def _lane_score(i):
        """Higher = more desirable. Green counts +1, danger −2, distance −0.1 per lane."""
        return len(lane_green[i]) - len(lane_danger[i]) * 2 - abs(i - car_lane) * 0.1

    tap_value    = 0.0
    tap_duration = 0.0

    if trailing:
        # Dodge AWAY from the side with most danger; toward the side with most green.
        if valid_danger:
            avg_dx = sum((_contour_cx(c) or center_x) for c in valid_danger) / len(valid_danger)
            tap_value = -1.0 if avg_dx > center_x else 1.0
        elif valid_green:
            green_xs  = [_contour_cx(c) for c in valid_green if _contour_cx(c) is not None]
            avg_gx    = sum(green_xs) / len(green_xs) if green_xs else center_x
            tap_value = 1.0 if avg_gx > center_x else -1.0
        else:
            tap_value = 1.0
        tap_duration = TRAILING_TAP

    elif p_active and not p_red_picked and red_target is not None:
        # Police challenge: steer toward the red token regardless of lane
        red_cx = _contour_cx(red_target)
        if red_cx is not None:
            offset = red_cx - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP

    elif not current_safe:
        # Current lane has danger → switch to the highest-scoring lane
        best_lane = max(range(n_lanes), key=_lane_score)
        if best_lane != car_lane:
            l_x, r_x = lane_bounds[best_lane]
            offset    = ((l_x + r_x) // 2) - center_x
            tap_value = float(np.clip(offset / (w * 0.35), -1.0, 1.0))
            if abs(tap_value) < 0.5:
                tap_value = 0.5 * np.sign(tap_value)
            tap_duration = DANGER_TAP

    elif current_green:
        # Current lane is safe AND has green → fine-tune toward the best green token,
        # ignoring any danger tokens that are in other lanes.
        best_green = max(lane_green[car_lane], key=cv2.contourArea)
        gx = _contour_cx(best_green)
        if gx is not None:
            offset = gx - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP

    else:
        # Current lane is clear but no green → move to nearest safe lane that has green
        safe_green_lanes = [i for i in range(n_lanes)
                            if not lane_danger[i] and lane_green[i]]
        if safe_green_lanes:
            best_lane = min(safe_green_lanes, key=lambda i: abs(i - car_lane))
            l_x, r_x  = lane_bounds[best_lane]
            offset    = ((l_x + r_x) // 2) - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP
        elif abs(road_offset) > w * 0.08:
            tap_value    = float(np.clip(road_offset / (w * 0.3), -0.5, 0.5))
            tap_duration = GREEN_TAP

    # Yellow disruption: dampen steering for stability during corrupted / delayed input
    if yellow_effect and abs(tap_value) > 0.0:
        tap_value *= 0.55

    # --- Debug overlay ---
    with data_lock:
        speed_mult   = challenge_state['speed_multiplier']
        p_active_v   = challenge_state['police_active']
        low_light_v  = challenge_state['low_light_active']
        yellow_eff_v = challenge_state['yellow_effect_active']

    debug = front_frame.copy()
    # Draw lane boundaries and highlight car's current lane
    for lx, rx in lane_bounds:
        cv2.line(debug, (lx, int(h * 0.45)), (lx, h), (200, 200, 0), 1)
    cv2.line(debug, (lane_bounds[-1][1], int(h * 0.45)),
             (lane_bounds[-1][1], h), (200, 200, 0), 1)
    cl_x, cl_rx = lane_bounds[car_lane]
    cv2.rectangle(debug, (cl_x, int(h * 0.72)), (cl_rx, h - 2), (0, 200, 200), 2)
    for c in valid_green:
        cv2.drawContours(debug, [c], -1, (0, 255, 0), 2)
    for c in valid_danger:
        cv2.drawContours(debug, [c], -1, (0, 0, 255), 2)
    for c in red_targets:
        cv2.drawContours(debug, [c], -1, (255, 0, 255), 2)
    cv2.putText(debug,
                f"Tap:{tap_value:+.2f}  Lane:{car_lane}/{n_lanes-1}  Safe:{current_safe}  G:{current_green}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug,
                f"Speed:{speed_mult:.2f}  Police:{p_active_v}  LL:{low_light_v}  YFX:{yellow_eff_v}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
        back_frame    = shared_data['latest_back_frame']
        prev_trailing = auto_state['trailing_car_detected']

    if back_frame is None:
        return

    h, w = back_frame.shape[:2]
    # Focus on center-lower region where a close trailing car appears large
    roi = back_frame[h // 2:, w // 4: 3 * w // 4]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Cars have saturated colors; exclude road gray, sky blue, and grass green
    car_mask  = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    not_blue  = cv2.bitwise_not(cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])))
    not_green = cv2.bitwise_not(cv2.inRange(hsv, np.array([35,  40, 40]), np.array([85,  255, 255])))
    car_mask  = cv2.bitwise_and(car_mask, cv2.bitwise_and(not_blue, not_green))

    car_pixels = int(np.sum(car_mask > 0))
    trailing   = car_pixels > 3000

    with data_lock:
        auto_state['trailing_car_detected'] = trailing

    # -----------------------------------------------------------------------
    # Challenge 2: Chasing Car — two timed chases (10s first, 3s second)
    # Rising edge starts a chase; car disappearing = avoided; timeout = penalty
    # -----------------------------------------------------------------------
    now          = time.time()
    chase_limits = (10.0, 3.0)

    with data_lock:
        chase_count  = challenge_state['chase_count']
        chase_active = challenge_state['chase_active']
        chase_start  = challenge_state['chase_start_time']

    if trailing and not prev_trailing and not chase_active and chase_count < 2:
        with data_lock:
            challenge_state['chase_active']     = True
            challenge_state['chase_start_time'] = now
            print(f"[C2] Chase {chase_count + 1} started! Limit: {chase_limits[chase_count]:.0f}s")

    elif chase_active:
        limit   = chase_limits[min(chase_count, 1)]
        elapsed = now - chase_start

        if not trailing:
            with data_lock:
                challenge_state['chase_active'] = False
                challenge_state['chase_count']  = min(chase_count + 1, 2)
                print(f"[C2] Chase {chase_count + 1} avoided!")
        elif elapsed > limit:
            with data_lock:
                challenge_state['chase_active']     = False
                challenge_state['chase_count']      = min(chase_count + 1, 2)
                challenge_state['speed_multiplier'] *= 0.5
                print(f"[C2] Chase {chase_count + 1} failed! "
                      f"Speed×0.5 → {challenge_state['speed_multiplier']:.2f}")


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
        steering     = shared_data['steering_input']
        acceleration = shared_data['acceleration_input']
        game_over    = challenge_state['game_over']
        speed_mult   = challenge_state['speed_multiplier']
        low_light    = challenge_state['low_light_active']

    if game_over:
        steering     = 0.0
        acceleration = -1.0
    elif acceleration > 0:
        # C1: temporary -10% while low light is active; stacks with permanent multiplier
        acceleration *= speed_mult * (0.9 if low_light else 1.0)

    try:
        data = struct.pack('ff', steering, acceleration)
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
