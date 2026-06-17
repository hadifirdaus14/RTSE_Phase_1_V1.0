import cv2
import numpy as np


def _contour_cx(contour):
    M = cv2.moments(contour)
    return int(M['m10'] / M['m00']) if M['m00'] > 0 else None


def _detect_tokens(frame, h, start_frac=0.45, end_frac=0.90):
    """
    Scan a window from start_frac to end_frac of frame height.
    Returns (valid_green, valid_danger) where danger = red + yellow.
    """
    roi    = cv2.cvtColor(frame[int(h * start_frac):int(h * end_frac), :], cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    green_mask = cv2.morphologyEx(
        cv2.inRange(roi, np.array([42, 80, 80]), np.array([78, 255, 255])),
        cv2.MORPH_OPEN, kernel
    )
    red_mask = cv2.morphologyEx(
        cv2.bitwise_or(
            cv2.inRange(roi, np.array([0,   100, 100]), np.array([8,   255, 255])),
            cv2.inRange(roi, np.array([172, 100, 100]), np.array([180, 255, 255]))
        ),
        cv2.MORPH_OPEN, kernel
    )
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
    lower    = cv2.cvtColor(frame[2 * h // 3:, :], cv2.COLOR_BGR2HSV)
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
    roi  = frame[int(h * 0.30):int(h * 0.75), :]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([100, 120, 120]), np.array([130, 255, 255])),
        cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    visible = int(np.sum(blue > 0)) > 300

    lower_hsv  = cv2.cvtColor(frame[h // 2:, :], cv2.COLOR_BGR2HSV)
    lower_blue = cv2.inRange(lower_hsv, np.array([100, 120, 120]), np.array([130, 255, 255]))
    collision  = int(np.sum(lower_blue > 0)) > 6000

    return visible, collision


def _detect_red_tokens_only(frame, h, start_frac=0.45, end_frac=0.90):
    """Return red-only contours in the standard scanning window."""
    roi    = cv2.cvtColor(frame[int(h * start_frac):int(h * end_frac), :], cv2.COLOR_BGR2HSV)
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
    roi      = cv2.cvtColor(frame[int(h * 0.80):, :], cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(roi, np.array([0,   100, 100]), np.array([8,   255, 255])),
        cv2.inRange(roi, np.array([172, 100, 100]), np.array([180, 255, 255]))
    )
    return int(np.sum(red_mask > 0)) > 500


def _yellow_token_in_close_range(frame, h):
    """Returns True when a yellow token is about to be hit (close-range bottom band)."""
    roi         = cv2.cvtColor(frame[int(h * 0.80):, :], cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.morphologyEx(
        cv2.inRange(roi, np.array([22, 100, 100]), np.array([33, 255, 255])),
        cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    return int(np.sum(yellow_mask > 0)) > 500


def _get_lane_info(frame, h, w, n_lanes=3):
    """
    Detect road lane boundaries from red barriers at the lower frame region.
    Returns:
        car_lane   – 0-based lane index (0 = leftmost, n_lanes-1 = rightmost)
        lane_bounds – list of (left_x, right_x) pixel ranges for every lane
    """
    lower    = cv2.cvtColor(frame[int(h * 0.65):, :], cv2.COLOR_BGR2HSV)
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
        if xs[0] < w // 2:
            road_left  = xs[0]
            road_right = min(xs[0] + road_width_est, w - 1)
        else:
            road_right = xs[0]
            road_left  = max(xs[0] - road_width_est, 0)
    else:
        road_left  = int(w * 0.10)
        road_right = int(w * 0.90)

    road_width = max(road_right - road_left, 1)
    lane_width = road_width / n_lanes
    car_pos    = (w / 2) - road_left
    car_lane   = int(np.clip(car_pos / lane_width, 0, n_lanes - 1))
    lane_bounds = [
        (int(road_left + i * lane_width), int(road_left + (i + 1) * lane_width))
        for i in range(n_lanes)
    ]
    return car_lane, lane_bounds


def _assign_lane(contour, lane_bounds):
    """Return 0-based lane index for a contour's horizontal centre."""
    cx = _contour_cx(contour)
    if cx is None:
        return -1
    for i, (lx, rx) in enumerate(lane_bounds):
        if lx <= cx < rx:
            return i
    return 0 if cx < lane_bounds[0][0] else len(lane_bounds) - 1
