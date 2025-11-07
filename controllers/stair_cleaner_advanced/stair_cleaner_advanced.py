# stair_cleaner_advanced.py
# Advanced AI controller: SLAM (occupancy grid), A* planning, camera dirt detection,
# path-following PID, task manager: find dirt -> plan -> go -> clean -> repeat.

import time
import math
import os
from controller import Robot, Motor, DistanceSensor, Camera, Lidar, InertialUnit
import numpy as np

from planner_utils import (world_to_map, map_to_world, raycast_grid,
                           astar, save_map_png, mark_free_line)

# ------------------- Config -------------------
TIME_STEP = 32
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# devices (names must match PROTO)
left_motor = robot.getDevice("left_motor")
right_motor = robot.getDevice("right_motor")
brush_motor = robot.getDevice("cleaner_motor")

front_ds = robot.getDevice("front_distance")           # forward
bottom_ds = robot.getDevice("bottom_distance")         # downward
imu = robot.getDevice("imu")
lidar = robot.getDevice("lidar")
camera = robot.getDevice("front_cam")

# enable devices
for d in (front_ds, bottom_ds, imu, lidar, camera):
    if d:
        try:
            d.enable(TIME_STEP)
        except Exception:
            pass

# set infinite rotation for wheels & brush
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
brush_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
brush_motor.setVelocity(0.0)

# ------------------- map params -------------------
MAP_RESOLUTION = 0.05       # meters per cell (5cm) — adjust for performance
MAP_SIZE_M = 10.0           # map covers MAP_SIZE_M x MAP_SIZE_M meters
MAP_CELLS = int(MAP_SIZE_M / MAP_RESOLUTION)
# map origin at lower-left; choose origin so robot near center initially
map_origin = (-MAP_SIZE_M/2.0, -MAP_SIZE_M/2.0)  # (ox, oy)
occ_map = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.float32)  # occupancy prob [0..1]

# robot pose estimate (x,y,theta) world coordinates (x->world x, y->world z)
pose_x = 0.0
pose_y = 0.0
pose_theta = 0.0

# odometry helpers (we'll estimate using wheel velocities; for more accuracy use encoders)
WHEEL_RADIUS = 0.04        # adjust to your robot
WHEEL_BASE = 0.20          # adjust to your robot (distance between wheels)
prev_left_vel = 0.0
prev_right_vel = 0.0

# ------------------- behavior params -------------------
MAX_SPEED = 4.0
TURN_SPEED = 1.5
ACCEL = 0.05

# planner / task state
task = "EXPLORE"   # tasks: EXPLORE, NAVIGATE_TO_DIRT, CLEANING, AVOID
current_path = None  # list of (row,col)
path_idx = 0

# dirt memory: list of (map_row,map_col)
dirt_cells = set()

# timing / logging
last_map_save = time.time()
MAP_SAVE_INTERVAL = 5.0

# PID for path following
kp_lin = 4.0
kd_lin = 0.5
prev_lin_err = 0.0
ki_lin = 0.0
int_lin = 0.0

kp_ang = 3.0
kd_ang = 0.2
prev_ang_err = 0.0
int_ang = 0.0

# helper: clamp
def clamp(v,a,b): return max(a,min(b,v))

# ------------------- low-level controls -------------------
current_speed = 0.0
def set_wheel_vel(vl, vr):
    left_motor.setVelocity(vl)
    right_motor.setVelocity(vr)

def ramp_drive(target_speed):
    global current_speed
    if current_speed < target_speed:
        current_speed = min(target_speed, current_speed + ACCEL)
    else:
        current_speed = max(target_speed, current_speed - ACCEL)
    set_wheel_vel(current_speed, current_speed)

# ------------------- sensor wrappers -------------------
def get_lidar_ranges():
    try:
        return lidar.getRangeImage()
    except Exception:
        return []

def get_camera_image():
    try:
        return camera.getImage()
    except Exception:
        return None

def get_imu_pitch():
    try:
        return imu.getRollPitchYaw()[1]
    except Exception:
        return 0.0

def get_front_distance():
    try:
        return front_ds.getValue()
    except Exception:
        return None

def get_down_distance():
    try:
        return bottom_ds.getValue()
    except Exception:
        return None

# ------------------- SLAM / mapping -------------------
def integrate_scan_into_map():
    # get lidar ranges and angles (Webots Lidar returns ranges in order across FOV)
    ranges = get_lidar_ranges()
    if len(ranges) == 0:
        return
    # approximate angle per index
    n = len(ranges)
    # lidar probably spans ~2*pi; we map indices to angles -pi..pi
    angles = np.linspace(-math.pi, math.pi, n)
    # convert robot pose to world XY
    rx, ry, th = pose_x, pose_y, pose_theta
    # build beam endpoints in world and mark free/occupied
    # We'll call simple raycast per-beam
    rows, cols = occ_map.shape
    for i, r in enumerate(ranges):
        if not np.isfinite(r) or r <= 0.0:
            continue
        angle = th + angles[i]
        ex = rx + r * math.cos(angle)
        ey = ry + r * math.sin(angle)
        mark_free_line(occ_map, (rx, ry), (ex, ey), map_origin, MAP_RESOLUTION)
        # endpoint occupied bump
        rr, cc = world_to_map(ex, ey, map_origin, MAP_RESOLUTION)
        if 0 <= rr < rows and 0 <= cc < cols:
            occ_map[rr, cc] = min(1.0, occ_map[rr, cc] + 0.8)

# ------------------- dirt detection (camera) -------------------
def detect_dirt_camera():
    # Try using OpenCV if available for speed; otherwise use Webots helpers
    img = get_camera_image()
    if img is None:
        return None
    w = camera.getWidth(); h = camera.getHeight()
    try:
        import cv2
        # convert Webots BGRA to numpy BGR (Webots returns bytes)
        buf = memoryview(img)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        bgr = arr[:, :, :3][:, :, ::-1]  # BGRA -> BGR
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # tune this to your dirt color; here we pick brown / dark
        lower = np.array([5, 50, 20])
        upper = np.array([30, 255, 200])
        mask = cv2.inRange(hsv, lower, upper)
        # find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) == 0:
            return None
        # pick largest contour
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            return None
        # compute centroid in image coords
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        # convert pixel to bearing: assume camera FOV symmetric
        fov = camera.getFieldOfView()
        bearing = (cx - w/2.0) / (w/2.0) * (fov/2.0)
        # estimate range crude using vertical position (nearer objects appear lower) -> heuristic
        est_range = 0.8  # fixed fallback
        return (bearing, est_range, cx, cy)
    except Exception:
        # fallback: center pixel brightness heuristic
        cx = w//2; cy = h//2
        r = Camera.imageGetRed(img, w, cx, cy)
        g = Camera.imageGetGreen(img, w, cx, cy)
        b = Camera.imageGetBlue(img, w, cx, cy)
        # brown-ish check
        if r > 100 and g < 100 and b < 80:
            # estimate bearing 0 and range 0.8
            return (0.0, 0.8, cx, cy)
        return None

# ------------------- A* TARGETING -------------------
def find_nearest_dirty_cell():
    if not dirt_cells:
        return None
    # compute robot map cell
    r0, c0 = world_to_map(pose_x, pose_y, map_origin, MAP_RESOLUTION)
    best = None; bestd = 1e9; best_cell = None
    for (dr, dc) in dirt_cells:
        d = math.hypot(dr - r0, dc - c0)
        if d < bestd:
            bestd = d
            best = (dr, dc)
    return best

def plan_path_to_cell(cell):
    r0, c0 = world_to_map(pose_x, pose_y, map_origin, MAP_RESOLUTION)
    path = astar(occ_map, (r0,c0), cell, obstacle_threshold=0.55)
    return path

# ------------------- path follower -------------------
def follow_path(path):
    global path_idx, prev_lin_err, int_lin, prev_ang_err, int_ang
    if path is None or len(path) == 0:
        return True
    # find goal cell world target
    gr, gc = path[path_idx]
    gx, gy = map_to_world(gr, gc, map_origin, MAP_RESOLUTION)
    # compute error in robot frame
    dx = gx - pose_x; dy = gy - pose_y
    target_dist = math.hypot(dx, dy)
    target_ang = math.atan2(dy, dx)
    ang_err = (target_ang - pose_theta + math.pi) % (2*math.pi) - math.pi
    # linear PID
    lin_err = target_dist
    int_lin += lin_err * (TIME_STEP/1000.0)
    deriv_lin = (lin_err - prev_lin_err) / (TIME_STEP/1000.0)
    prev_lin_err = lin_err
    v = kp_lin * lin_err + ki_lin * int_lin + kd_lin * deriv_lin
    # angular PID
    int_ang += ang_err * (TIME_STEP/1000.0)
    deriv_ang = (ang_err - prev_ang_err) / (TIME_STEP/1000.0)
    prev_ang_err = ang_err
    w = kp_ang * ang_err + kd_ang * deriv_ang
    # convert to wheel velocities
    vl = v - (w * WHEEL_BASE/2.0) / WHEEL_RADIUS
    vr = v + (w * WHEEL_BASE/2.0) / WHEEL_RADIUS
    # clamp
    vl = clamp(vl, -MAX_SPEED, MAX_SPEED)
    vr = clamp(vr, -MAX_SPEED, MAX_SPEED)
    set_wheel_vel(vl, vr)
    # if reached cell, advance
    if target_dist < 0.1:
        path_idx += 1
        if path_idx >= len(path):
            return True
    return False

# ------------------- main loop -------------------
start_time = time.time()
step_counter = 0
# main
while robot.step(TIME_STEP) != -1:
    step_counter += 1
    # ----- 1) update odometry (using wheel velocities) -----
    # approximate from motor velocities (rad/s) -> linear velocity
    try:
        vl = left_motor.getVelocity()
        vr = right_motor.getVelocity()
    except Exception:
        vl = vr = 0.0
    v_lin = WHEEL_RADIUS * 0.5 * (vl + vr)
    omega = WHEEL_RADIUS * (vr - vl) / WHEEL_BASE
    dt = TIME_STEP / 1000.0
    # integrate pose
    pose_x += v_lin * math.cos(pose_theta) * dt
    pose_y += v_lin * math.sin(pose_theta) * dt
    pose_theta += omega * dt
    # normalize
    pose_theta = (pose_theta + math.pi) % (2*math.pi) - math.pi

    # ----- 2) integrate lidar into map periodically -----
    if step_counter % 1 == 0:
        ranges = get_lidar_ranges()
        if len(ranges) > 0:
            # convert to angles and call planner_utils.raycast_grid
            n = len(ranges)
            angles = np.linspace(-math.pi, math.pi, n)
            rows, cols = occ_map.shape
            rx, ry = pose_x, pose_y
            for i, r in enumerate(ranges):
                if not np.isfinite(r) or r <= 0.0:
                    continue
                ang = pose_theta + angles[i]
                ex = rx + r * math.cos(ang)
                ey = ry + r * math.sin(ang)
                mark_free_line(occ_map, (rx, ry), (ex, ey), map_origin, MAP_RESOLUTION)
                rr, cc = world_to_map(ex, ey, map_origin, MAP_RESOLUTION)
                if 0 <= rr < rows and 0 <= cc < cols:
                    occ_map[rr, cc] = min(1.0, occ_map[rr, cc] + 0.65)
    # ----- 3) camera dirt detection -----
    if step_counter % 3 == 0:
        det = detect_dirt_camera()
        if det:
            bearing, r_est, cx, cy = det
            # compute world coordinate of detected dirt
            wx = pose_x + r_est * math.cos(pose_theta + bearing)
            wy = pose_y + r_est * math.sin(pose_theta + bearing)
            dr, dc = world_to_map(wx, wy, map_origin, MAP_RESOLUTION)
            if 0 <= dr < occ_map.shape[0] and 0 <= dc < occ_map.shape[1]:
                dirt_cells.add((dr, dc))
                # lower occupancy around dirt (so planner can plan onto it)
                for rr in range(max(0,dr-1), min(occ_map.shape[0], dr+2)):
                    for cc in range(max(0,dc-1), min(occ_map.shape[1], dc+2)):
                        occ_map[rr,cc] = max(0.0, occ_map[rr,cc] - 0.6)
    # ----- 4) periodic map save -----
    if time.time() - last_map_save > MAP_SAVE_INTERVAL:
        try:
            save_map_png(occ_map, "/tmp/stair_map.png")
        except Exception:
            pass
        last_map_save = time.time()

    # ----- 5) decide task -----
    if task == "EXPLORE":
        # if we have known dirt, go to it
        if dirt_cells:
            target = find_nearest_dirty_cell()
            if target:
                path = plan_path_to_cell(target)
                if path:
                    current_path = path
                    path_idx = 0
                    task = "NAVIGATE_TO_DIRT"
                else:
                    # can't find path -> remove dirt candidate (maybe unreachable)
                    dirt_cells.remove(target)
        else:
            # exploration behavior: simple forward, avoid obstacles
            # check front distance (use raw front_ds)
            fd = get_front_distance()
            if fd is not None and fd < 0.5:
                # turn a bit
                left_motor.setVelocity(-1.0); right_motor.setVelocity(1.0)
                robot.step(int(200))
            else:
                # go forward
                left_motor.setVelocity(2.0); right_motor.setVelocity(2.0)
    elif task == "NAVIGATE_TO_DIRT":
        if current_path is None:
            task = "EXPLORE"
        else:
            done = follow_path(current_path)
            if done:
                # begin cleaning procedure
                task = "CLEANING"
                left_motor.setVelocity(0); right_motor.setVelocity(0)
                brush_motor.setVelocity(20.0)
                clean_start = time.time()
    elif task == "CLEANING":
        # run brush for a few seconds while staying put
        if time.time() - clean_start > 4.0:
            brush_motor.setVelocity(0)
            # mark dirt cell removed
            # remove nearest dirt cell
            if dirt_cells:
                nearest = find_nearest_dirty_cell()
                if nearest:
                    try:
                        dirt_cells.remove(nearest)
                    except KeyError:
                        pass
            task = "EXPLORE"
    elif task == "AVOID":
        # simple avoidance then back to explore
        left_motor.setVelocity(-1.5); right_motor.setVelocity(-1.5)
        robot.step(200)
        left_motor.setVelocity(1.5); right_motor.setVelocity(-1.5)
        robot.step(300)
        task = "EXPLORE"

# end main loop
