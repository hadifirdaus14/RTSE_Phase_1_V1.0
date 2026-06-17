import threading

# Network constants
CAMERA_HOST       = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT  = 8082
CONTROL_HOST      = '127.0.0.1'
CONTROL_PORT      = 8081

# Shared frame/control data (protected by data_lock)
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame':  None,
    'steering_input':     0.0,
    'acceleration_input': 0.0,
}
data_lock = threading.Lock()

# Mutable global flags / sockets — always access via `import shared as _sh; _sh.xxx`
is_running       = True
front_camera_sock = None
back_camera_sock  = None
control_conn      = None
