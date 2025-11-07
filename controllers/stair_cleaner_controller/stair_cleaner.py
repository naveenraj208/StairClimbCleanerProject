"""
stair_cleaner.py
Combined, cleaned and ready-to-run controller for:
- wheel control (smooth acceleration)
- stair detection & climbing (using front/down distance sensors + IMU)
- basic PID pitch regulation while climbing
- LIDAR obstacle stop
- camera access (no heavy CV included)
- rotating brush control and simple dirt detection counter

Place this file in: controllers/stair_cleaner/stair_cleaner.py
"""

from controller import Robot, Camera
import math
import sys
import time

# ---------------------------
# Helper: robust device getter
# ---------------------------
def get_device(robot, possible_names):
    for n in possible_names:
        try:
            dev = robot.getDevice(n)
            if dev:
                return dev, n
        except Exception:
            continue
    return None, None

# ---------------------------
# Initialization
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# motors (try several common names)
left_motor, left_name = get_device(robot, ["left_motor", "left_wheel_motor", "leftMotor"])
right_motor, right_name = get_device(robot, ["right_motor", "right_wheel_motor", "rightMotor"])
brush_motor, brush_name = get_device(robot, ["brush_motor", "cleaner_motor", "brushMotor"])
flipper_motor, flipper_name = get_device(robot, ["flipper_motor", "flipper_left_motor"])  # optional

if not left_motor or not right_motor:
    print("ERROR: left or right motor not found. Check your PROTO device names.")
    sys.exit(1)

# set continuous rotation for wheel motors
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# brush motor if available
if brush_motor:
    brush_motor.setPosition(float("inf"))
    brush_motor.setVelocity(0.0)

# flipper motor (optional)
if flipper_motor:
    flipper_motor.setPosition(float("inf"))
    flipper_motor.setVelocity(0.0)

# sensors: distance front / down
front_ds, _ = get_device(robot, ["front_ds", "front_distance", "front_dist", "frontSensor"])
down_ds, _ = get_device(robot, ["down_ds", "down_distance", "bottom_distance", "downSensor"])

# dirt sensor (optional)
dirt_sensor, _ = get_device(robot, ["dirt_sensor", "dirtSensor"])

# IMU
imu, _ = get_device(robot, ["imu", "inertial_unit", "inertialUnit"])
if imu:
    imu.enable(timestep)

# LIDAR (optional)
lidar, _ = get_device(robot, ["lidar", "Lidar"])
if lidar:
    try:
        lidar.enable(timestep)
        # some Webots versions require this to get point cloud
        lidar.enablePointCloud()
    except Exception:
        pass

# Camera (optional)
camera, _ = get_device(robot, ["front_cam", "camera", "frontCamera"])
if camera:
    camera.enable(timestep)

# enable distance sensors if present
if front_ds:
    try:
        front_ds.enable(timestep)
    except Exception:
        pass
if down_ds:
    try:
        down_ds.enable(timestep)
    except Exception:
        pass
if dirt_sensor:
    try:
        dirt_sensor.enable(timestep)
    except Exception:
        pass

# ---------------------------
# Parameters / tunables
# ---------------------------
# thresholds (may require tuning for your world)
FRONT_DETECT = 700        # sensor raw threshold for front obstacle (smaller => closer)
DOWN_DROP = 900           # sensor raw threshold for downward drop (larger => drop)
DIRT_THRESHOLD = 300      # dirt sensor threshold (example)
LIDAR_OBSTACLE_DIST = 0.3 # meters

# motion params
MAX_SPEED = 4.0           # default forward wheel speed (rad/s)
ACCEL_RATE = 0.02         # ramping
CLIMB_MIN_SPEED = 1.2     # minimum speed during climbing
CLIMB_MAX_SPEED = 6.0     # clamp climbing speed

# wheel geometry (used only for FK monitoring)
WHEEL_RADIUS = 0.035      # metres (adjust to your robot)
WHEEL_DISTANCE = 0.18     # metres between wheels (adjust)

# PID for pitch regulation while climbing
KP = 2.0
KI = 0.0
KD = 0.4
prev_error = 0.0
integral = 0.0

# safety
MAX_SAFE_PITCH = 0.6      # radians (~34 deg) — if higher, slow down

# brush
BRUSH_ON_SPEED = 15.0
BRUSH_OFF_SPEED = 0.0

# state machine
STATE_MOVE = "MOVE_FORWARD"
STATE_CLIMB = "CLIMB"
STATE_STABILIZE = "STABILIZE"
STATE_BACK = "BACK"

state = STATE_MOVE
step_count = 0
cleaned_count = 0

# internal drive tracker for smooth acceleration
current_speed = 0.0

# helper: read sensor value safely
def sensor_value(dev):
    try:
        return dev.getValue()
    except Exception:
        return None

# helper: lidar front check
def lidar_obstacle():
    if not lidar:
        return False
    try:
        ranges = lidar.getRangeImage()
        # index 0 in many setups = front; but to be safe, index mid
        if len(ranges) == 0:
            return False
        # pick center
        mid = len(ranges) // 2
        front = ranges[mid]
        return front < LIDAR_OBSTACLE_DIST
    except Exception:
        return False

# helper: camera center pixel (returns (r,g,b) 0-255)
def camera_center_pixel():
    if not camera:
        return None
    try:
        img = camera.getImage()
        w = camera.getWidth()
        h = camera.getHeight()
        cx = w // 2
        cy = h // 2
        # Camera.imageGet* are static methods on Camera class in python API
        r = Camera.imageGetRed(img, w, cx, cy)
        g = Camera.imageGetGreen(img, w, cx, cy)
        b = Camera.imageGetBlue(img, w, cx, cy)
        return (r, g, b)
    except Exception:
        return None

# smooth drive towards a target velocity
def smooth_drive(target):
    global current_speed
    if current_speed < target:
        current_speed += ACCEL_RATE
        if current_speed > target:
            current_speed = target
    elif current_speed > target:
        current_speed -= ACCEL_RATE
        if current_speed < target:
            current_speed = target
    left_motor.setVelocity(current_speed)
    right_motor.setVelocity(current_speed)

# simple step detection using front and down sensors
def detect_step():
    front_val = sensor_value(front_ds) if front_ds else None
    down_val = sensor_value(down_ds) if down_ds else None
    is_step = False
    is_edge = False
    # these thresholds are approximate; tune them for your world
    if front_val is not None:
        is_step = (front_val < FRONT_DETECT)
    if down_val is not None:
        is_edge = (down_val > DOWN_DROP)
    return is_step, is_edge, front_val, down_val

# dirt detect (no world removal here, just counts)
def detect_dirt():
    if not dirt_sensor:
        return False
    val = sensor_value(dirt_sensor)
    if val is None:
        return False
    return val < DIRT_THRESHOLD

# small helper to clamp speed
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

print("=== Stair Cleaner controller started ===")
print("Detected devices:")
print(" left motor:", left_name)
print(" right motor:", right_name)
print(" brush motor:", brush_name)
print(" flipper motor:", flipper_name)
print(" front dist:", getattr(front_ds, 'getName', lambda: None)())
print(" down dist:", getattr(down_ds, 'getName', lambda: None)())
print(" imu:", getattr(imu, 'getName', lambda: None)())
print(" lidar:", getattr(lidar, 'getName', lambda: None)())
print(" camera:", getattr(camera, 'getName', lambda: None)())
print(" dirt sensor:", getattr(dirt_sensor, 'getName', lambda: None)())

# ---------------------------
# MAIN LOOP
# ---------------------------
while robot.step(timestep) != -1:
    # get sensors
    step_ahead, edge, front_val, down_val = detect_step()
    pitch = None
    if imu:
        try:
            pitch = imu.getRollPitchYaw()[1]
        except Exception:
            pitch = None

    # LIDAR obstacle check (prevents collision)
    if lidar_obstacle():
        # stop smoothly
        smooth_drive(0.0)
        if brush_motor:
            brush_motor.setVelocity(BRUSH_OFF_SPEED)
        # skip rest of logic this step
        # print debug
        # print("[LIDAR] obstacle detected — stopped")
        continue

    # simple dirt detection: increment counter and run brush
    if detect_dirt():
        cleaned_count += 1
        print(f"[DIRT] Detected & counted. Total cleaned: {cleaned_count}")
        # spin brush a bit
        if brush_motor:
            brush_motor.setVelocity(BRUSH_ON_SPEED)
    # otherwise brush runs only when moving
    # state machine
    if state == STATE_MOVE:
        # drive forward at cruising speed
        smooth_drive(MAX_SPEED)
        # turn brush on when moving
        if brush_motor:
            brush_motor.setVelocity(BRUSH_ON_SPEED if current_speed > 0.05 else BRUSH_OFF_SPEED)

        # check for step or edge
        if step_ahead:
            print("[STATE] step ahead -> CLIMB")
            state = STATE_CLIMB
        elif edge:
            print("[STATE] edge detected -> BACK")
            state = STATE_BACK

    elif state == STATE_CLIMB:
        # climb control using PID on pitch (if IMU available)
        # set a target_pitch (slight forward lean)
        target_pitch = 0.18  # radians, tweak for your robot/world

        if pitch is None:
            # no IMU: simple push forward for fixed time
            left_motor.setVelocity(3.5)
            right_motor.setVelocity(3.5)
        else:
            # PID to regulate pitch towards target_pitch while moving
            error = target_pitch - pitch
            integral += error * (timestep / 1000.0)
            derivative = (error - prev_error) / (timestep / 1000.0)
            prev_error = error

            pid_out = KP * error + KI * integral + KD * derivative

            base = 4.0
            climb_speed = clamp(base + pid_out, CLIMB_MIN_SPEED, CLIMB_MAX_SPEED)
            left_motor.setVelocity(climb_speed)
            right_motor.setVelocity(climb_speed)

        # brush on while climbing
        if brush_motor:
            brush_motor.setVelocity(BRUSH_ON_SPEED)

        # detect stabilization (pitch returns near level) -> STABILIZE
        if pitch is not None and abs(pitch) < 0.12:
            print("[STATE] stabilized on top -> STABILIZE")
            state = STATE_STABILIZE

    elif state == STATE_STABILIZE:
        # stop and wait a short time, then continue
        smooth_drive(0.0)
        if brush_motor:
            brush_motor.setVelocity(BRUSH_ON_SPEED)
        # small settle
        robot.step(int(200))  # 200 ms pause
        step_count += 1
        print(f"[STATE] Step complete. Steps climbed: {step_count}")
        state = STATE_MOVE

    elif state == STATE_BACK:
        # back away from edge a bit then turn
        left_motor.setVelocity(-2.0)
        right_motor.setVelocity(-2.0)
        if brush_motor:
            brush_motor.setVelocity(BRUSH_OFF_SPEED)
        robot.step(int(300))
        # small turn to avoid same edge (spin right)
        left_motor.setVelocity(1.5)
        right_motor.setVelocity(-1.5)
        robot.step(int(400))
        state = STATE_MOVE

    # safety: if pitch dangerously high, slow down
    if pitch is not None and pitch > MAX_SAFE_PITCH:
        print("[SAFETY] high pitch detected — slowing down")
        left_motor.setVelocity(1.0)
        right_motor.setVelocity(1.0)
        if brush_motor:
            brush_motor.setVelocity(BRUSH_OFF_SPEED)

# end main loop
