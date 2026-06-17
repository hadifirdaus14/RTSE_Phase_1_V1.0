import shared as _sh

# Autonomous driving state (protected by data_lock)
auto_state = {
    'tap_end_time': 0.0,
    'trailing_car_detected': False,
}

challenge_state = {
    # Challenge 1: Low Light (once in first 10s)
    'low_light_active': False,
    'low_light_triggered': False,
    'low_light_start_time': 0.0,

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
    'yellow_effect_active': False,
    'yellow_effect_end_time': 0.0,

    # C1 brightness tracking
    # brightness_max: slow-decay peak brightness (rises instantly, falls at 0.1%/frame)
    # c1_dark_count : consecutive frames below threshold; 5 required to fire
    'brightness_max':  0.0,
    'c1_dark_count':   0,
    'c1_last_report':  0.0,

    # Permanent speed multiplier — stacks with each -50% penalty
    'speed_multiplier': 1.0,
}

# Reconnection tracking (mutable dict to avoid global-reassignment import issues)
reconnect_state = {
    'ctrl_reconnecting': False,
    'front_stale_since': 0.0,
    'back_stale_since': 0.0,
}


def _reset_session():
    """Reset all challenge and driving state for a new game session."""
    with _sh.data_lock:
        challenge_state.update({
            'low_light_active':       False,
            'low_light_triggered':    False,
            'low_light_start_time':   0.0,
            'chase_count':            0,
            'chase_active':           False,
            'chase_start_time':       0.0,
            'police_active':          False,
            'police_start_time':      0.0,
            'police_red_picked':      False,
            'police_done':            False,
            'game_over':              False,
            'yellow_effect_active':   False,
            'yellow_effect_end_time': 0.0,
            'brightness_max':          0.0,
            'c1_dark_count':          0,
            'c1_last_report':         0.0,
            'speed_multiplier':       1.0,
        })
        auto_state['tap_end_time']          = 0.0
        auto_state['trailing_car_detected'] = False
        _sh.shared_data['steering_input']     = 0.0
        _sh.shared_data['acceleration_input'] = 0.0
        _sh.shared_data['latest_front_frame'] = None
        _sh.shared_data['latest_back_frame']  = None
    reconnect_state['front_stale_since'] = 0.0
    reconnect_state['back_stale_since']  = 0.0
    print("[SESSION] State reset — ready for new game run.")
