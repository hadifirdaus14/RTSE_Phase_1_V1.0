import socket
import time
import struct
import threading
import cv2
import numpy as np

import shared as _sh
from state import auto_state, challenge_state, reconnect_state, _reset_session

_c1_log_time = 0.0  # throttle periodic brightness prints
from vision import (
    _contour_cx,
    _detect_tokens,
    _detect_road_offset,
    _detect_police_car,
    _detect_red_tokens_only,
    _yellow_token_in_close_range,
    _get_lane_info,
    _assign_lane,
)


def _reconnect_control():
    """Background thread: wait for the simulator to reconnect, then reset state."""
    print("[Control] Waiting for new game connection...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((_sh.CONTROL_HOST, _sh.CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    while _sh.is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"[Control] New game connected from {addr}")
            _sh.control_conn = conn
            break
        except socket.timeout:
            continue
    _reset_session()
    reconnect_state['ctrl_reconnecting'] = False
    print("[Control] New game session started.")


def processing_task():
    with _sh.data_lock:
        front_frame = _sh.shared_data['latest_front_frame']
        tap_end     = auto_state['tap_end_time']
        trailing    = auto_state['trailing_car_detected']
        game_over   = challenge_state['game_over']

    if front_frame is None:
        return

    now = time.time()

    if game_over:
        with _sh.data_lock:
            _sh.shared_data['acceleration_input'] = -1.0
            _sh.shared_data['steering_input']     = 0.0
        return

    h, w      = front_frame.shape[:2]
    center_x  = w // 2
    GREEN_TAP    = 0.05
    DANGER_TAP   = 0.05
    TRAILING_TAP = 0.25

    # ------------------------------------------------------------------
    # Challenge 1: Low Light — slow-decay peak + consecutive dark-frame counter.
    #
    # brightness_max tracks the recent peak brightness:
    #   - jumps UP instantly when a brighter frame arrives
    #   - decays DOWN at only 0.1 % per frame when it doesn't
    # This means brightness_max stays at 0.0 until the game scene actually
    # shows something bright.  Startup dark frames never raise it, so the
    # threshold (75 % of max) starts at 0 and cannot be crossed until
    # max > 60 — which eliminates startup false-positives entirely.
    #
    # During the actual C1 event brightness_max is frozen (not updated) so
    # the recovery threshold is anchored to the last pre-dark value.
    #
    # Five consecutive dark frames are required to fire (filters 1-4 frame
    # transients) and the counter re-builds in ~25 ms after any state reset.
    # ------------------------------------------------------------------
    global _c1_log_time
    with _sh.data_lock:
        c1_triggered = challenge_state['low_light_triggered']

    if not c1_triggered:
        gray           = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray))

        with _sh.data_lock:
            bmax       = challenge_state['brightness_max']
            dark_count = challenge_state['c1_dark_count']

            if not challenge_state['low_light_active']:
                # Update slow-decay peak (frozen during dark event)
                if avg_brightness >= bmax:
                    bmax = avg_brightness           # instant jump up
                else:
                    bmax = max(avg_brightness, bmax * 0.999)  # slow decay
                challenge_state['brightness_max'] = bmax

                # Periodic log every 2s
                if now - _c1_log_time >= 2.0:
                    _c1_log_time = now
                    print(f"[C1] brightness={avg_brightness:.1f}  peak={bmax:.1f}  dark_count={dark_count}")

                # Armed only once we've seen a genuinely bright scene (peak > 60).
                # Dark threshold: current < 75 % of peak brightness.
                is_dark = bmax > 60 and avg_brightness < bmax * 0.75
                if is_dark:
                    dark_count += 1
                    challenge_state['c1_dark_count'] = dark_count
                    if dark_count >= 5:
                        challenge_state['low_light_active']     = True
                        challenge_state['low_light_start_time'] = now
                        challenge_state['c1_last_report']       = now
                        print(f"[C1] Dark detected! brightness={avg_brightness:.1f}  "
                              f"peak={bmax:.1f} → sending acceleration=-1.0")
                else:
                    challenge_state['c1_dark_count'] = 0
            else:
                # brightness_max is frozen — anchored to pre-dark peak value
                # Recovery: current brightness ≥ 85 % of that peak
                recovered = avg_brightness >= bmax * 0.85
                if now - challenge_state['c1_last_report'] >= 1.0:
                    challenge_state['c1_last_report'] = now
                    print(f"[C1] Holding -1.0 | brightness={avg_brightness:.1f}  "
                          f"target≥{bmax * 0.85:.1f}")
                if recovered:
                    challenge_state['low_light_active']    = False
                    challenge_state['low_light_triggered'] = True
                    challenge_state['c1_dark_count']       = 0
                    print(f"[C1] Light recovered! brightness={avg_brightness:.1f}  "
                          f"peak={bmax:.1f}")

    with _sh.data_lock:
        low_light = challenge_state['low_light_active']

    # ------------------------------------------------------------------
    # Challenge 3: Police Car — detect blue lights, seek red token in 10s
    # ------------------------------------------------------------------
    police_visible, police_collision = _detect_police_car(front_frame, h, w)

    with _sh.data_lock:
        p_active     = challenge_state['police_active']
        p_done       = challenge_state['police_done']
        p_red_picked = challenge_state['police_red_picked']

    if police_visible and not p_active and not p_done and _sh.control_conn is not None:
        with _sh.data_lock:
            challenge_state['police_active']     = True
            challenge_state['police_start_time'] = now
            print("[C3] Police car spotted! Collect a red token within 10s.")
        p_active = True

    if p_active:
        if police_collision:
            with _sh.data_lock:
                challenge_state['game_over'] = True
                print("[C3] GAME OVER — collided with police car!")
            with _sh.data_lock:
                _sh.shared_data['acceleration_input'] = -1.0
                _sh.shared_data['steering_input']     = 0.0
            return

        with _sh.data_lock:
            p_start = challenge_state['police_start_time']

        if (now - p_start) >= 10.0 and not p_red_picked:
            with _sh.data_lock:
                challenge_state['police_active']     = False
                challenge_state['police_done']       = True
                challenge_state['speed_multiplier'] *= 0.5
                print(f"[C3] Timer expired — no red token! "
                      f"Speed×0.5 → {challenge_state['speed_multiplier']:.2f}")
            p_active = False
        elif p_red_picked:
            with _sh.data_lock:
                challenge_state['police_active'] = False
                challenge_state['police_done']   = True
                print("[C3] Red token collected — police challenge cleared!")
            p_active = False
        else:
            close_red = _detect_red_tokens_only(front_frame, h, start_frac=0.70, end_frac=0.95)
            token_red = [c for c in close_red if cv2.contourArea(c) > 800]
            if token_red:
                with _sh.data_lock:
                    challenge_state['police_red_picked'] = True
                    p_red_picked = True

    # ------------------------------------------------------------------
    # Yellow Token Effect — 5s conservative mode
    # ------------------------------------------------------------------
    if _yellow_token_in_close_range(front_frame, h):
        with _sh.data_lock:
            if not challenge_state['yellow_effect_active']:
                challenge_state['yellow_effect_active']   = True
                challenge_state['yellow_effect_end_time'] = now + 5.0
                print("[Yellow] Yellow token hit — 5s disruption mode active.")

    with _sh.data_lock:
        if challenge_state['yellow_effect_active'] and now > challenge_state['yellow_effect_end_time']:
            challenge_state['yellow_effect_active'] = False
            print("[Yellow] Yellow effect expired.")
        yellow_effect = challenge_state['yellow_effect_active']

    if now < tap_end and trailing:
        return

    valid_green, valid_danger = _detect_tokens(front_frame, h)
    red_targets = _detect_red_tokens_only(front_frame, h) if p_active else []
    road_offset = _detect_road_offset(front_frame, h, w)
    red_target  = max(red_targets, key=cv2.contourArea) if red_targets else None

    # ------------------------------------------------------------------
    # Lane-aware steering
    # Priority: trailing car > police red-seek > danger avoidance
    #           > green fine-tune > seek nearest safe green > road-centre
    # ------------------------------------------------------------------
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
        return len(lane_green[i]) - len(lane_danger[i]) * 2 - abs(i - car_lane) * 0.1

    tap_value    = 0.0
    tap_duration = 0.0

    if trailing:
        if valid_danger:
            avg_dx    = sum((_contour_cx(c) or center_x) for c in valid_danger) / len(valid_danger)
            tap_value = -1.0 if avg_dx > center_x else 1.0
        elif valid_green:
            green_xs  = [_contour_cx(c) for c in valid_green if _contour_cx(c) is not None]
            avg_gx    = sum(green_xs) / len(green_xs) if green_xs else center_x
            tap_value = 1.0 if avg_gx > center_x else -1.0
        else:
            tap_value = 1.0
        tap_duration = TRAILING_TAP

    elif p_active and not p_red_picked and red_target is not None:
        red_cx = _contour_cx(red_target)
        if red_cx is not None:
            offset = red_cx - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP

    elif not current_safe:
        best_lane = max(range(n_lanes), key=_lane_score)
        if best_lane != car_lane:
            l_x, r_x  = lane_bounds[best_lane]
            offset    = ((l_x + r_x) // 2) - center_x
            tap_value = float(np.clip(offset / (w * 0.35), -1.0, 1.0))
            if abs(tap_value) < 0.5:
                tap_value = 0.5 * np.sign(tap_value)
            tap_duration = DANGER_TAP

    elif current_green:
        best_green = max(lane_green[car_lane], key=cv2.contourArea)
        gx = _contour_cx(best_green)
        if gx is not None:
            offset = gx - center_x
            if abs(offset) > w * 0.03:
                tap_value    = float(np.clip(offset / center_x, -1.0, 1.0))
                tap_duration = GREEN_TAP

    else:
        safe_green_lanes = [i for i in range(n_lanes) if not lane_danger[i] and lane_green[i]]
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

    if yellow_effect and abs(tap_value) > 0.0:
        tap_value *= 0.55

    # --- Debug overlay ---
    with _sh.data_lock:
        speed_mult   = challenge_state['speed_multiplier']
        p_active_v   = challenge_state['police_active']
        low_light_v  = challenge_state['low_light_active']
        yellow_eff_v = challenge_state['yellow_effect_active']

    debug = front_frame.copy()
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

    with _sh.data_lock:
        _sh.shared_data['acceleration_input'] = 1.0
        if abs(tap_value) > 0.05:
            _sh.shared_data['steering_input'] = float(tap_value)
            auto_state['tap_end_time']        = now + tap_duration
        else:
            _sh.shared_data['steering_input'] = 0.0


def back_camera_processing_task():
    """Detect trailing cars and manage Challenge 2 (Chasing Car)."""
    with _sh.data_lock:
        back_frame    = _sh.shared_data['latest_back_frame']
        prev_trailing = auto_state['trailing_car_detected']

    if back_frame is None:
        return

    h, w = back_frame.shape[:2]
    roi  = back_frame[h // 2:, w // 4: 3 * w // 4]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    car_mask  = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    not_blue  = cv2.bitwise_not(cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])))
    not_green = cv2.bitwise_not(cv2.inRange(hsv, np.array([35,  40, 40]), np.array([85,  255, 255])))
    car_mask  = cv2.bitwise_and(car_mask, cv2.bitwise_and(not_blue, not_green))

    car_pixels = int(np.sum(car_mask > 0))
    trailing   = car_pixels > 1500

    with _sh.data_lock:
        auto_state['trailing_car_detected'] = trailing

    # ------------------------------------------------------------------
    # Challenge 2: Chasing Car — two timed chases (10s first, 3s second)
    # ------------------------------------------------------------------
    now          = time.time()
    chase_limits = (10.0, 3.0)

    with _sh.data_lock:
        chase_count  = challenge_state['chase_count']
        chase_active = challenge_state['chase_active']
        chase_start  = challenge_state['chase_start_time']

    if trailing and not prev_trailing and not chase_active and chase_count < 2:
        with _sh.data_lock:
            challenge_state['chase_active']     = True
            challenge_state['chase_start_time'] = now
            print(f"[C2] Chase {chase_count + 1} started! Limit: {chase_limits[chase_count]:.0f}s")

    elif chase_active:
        limit   = chase_limits[min(chase_count, 1)]
        elapsed = now - chase_start

        if not trailing:
            with _sh.data_lock:
                challenge_state['chase_active'] = False
                challenge_state['chase_count']  = min(chase_count + 1, 2)
                print(f"[C2] Chase {chase_count + 1} avoided!")
        elif elapsed > limit:
            with _sh.data_lock:
                challenge_state['chase_active']     = False
                challenge_state['chase_count']      = min(chase_count + 1, 2)
                challenge_state['speed_multiplier'] *= 0.5
                print(f"[C2] Chase {chase_count + 1} failed! "
                      f"Speed×0.5 → {challenge_state['speed_multiplier']:.2f}")


def send_controls_task():
    if _sh.control_conn is None:
        if not reconnect_state['ctrl_reconnecting']:
            reconnect_state['ctrl_reconnecting'] = True
            threading.Thread(target=_reconnect_control, daemon=True).start()
        return

    now = time.time()
    with _sh.data_lock:
        if now >= auto_state['tap_end_time']:
            _sh.shared_data['steering_input'] = 0.0
        steering       = _sh.shared_data['steering_input']
        acceleration   = _sh.shared_data['acceleration_input']
        game_over  = challenge_state['game_over']
        speed_mult = challenge_state['speed_multiplier']
        low_light  = challenge_state['low_light_active']

    if game_over:
        steering     = 0.0
        acceleration = -1.0
    elif low_light:
        # C1 spec: send -1.0 to recover the light.
        # Do NOT override steering — Chase 1 steering must still work in parallel.
        acceleration = -1.0
    elif acceleration > 0:
        acceleration *= speed_mult

    try:
        data = struct.pack('ff', steering, acceleration)
        _sh.control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        _sh.control_conn = None
