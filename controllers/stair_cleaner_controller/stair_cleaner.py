from controller import Robot

# ================================================================
#  Stair-Climbing Cleaner Robot – Webots Controller (Python)
#  Basic Logic:
#     1. Move forward
#     2. Detect stair edge
#     3. Climb up stair step-by-step using IMU angle detection
#     4. Continue cleaning on each level
# ================================================================

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ------------------------------------------------
# Get Devices
# ------------------------------------------------
left_motor = robot.getDevice("left_motor")
right_motor = robot.getDevice("right_motor")
cleaner = robot.getDevice("cleaner_motor")

front_sensor = robot.getDevice("front_distance")
bottom_sensor = robot.getDevice("bottom_distance")
imu = robot.getDevice("imu")

front_sensor.enable(timestep)
bottom_sensor.enable(timestep)
imu.enable(timestep)

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
cleaner.setPosition(float("inf"))

left_motor.setVelocity(0)
right_motor.setVelocity(0)
cleaner.setVelocity(3.0)   # always ON

# ------------------------------------------------
# Motion Helpers
# ------------------------------------------------
def forward(speed=4.0):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

def stop():
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def turn_left(speed=3):
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(speed)

def turn_right(speed=3):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(-speed)

def climb_step():
    """
    Simple stair climbing strategy:
    1. Move until front hits stair
    2. Increase speed to lift front wheel
    3. Wait until IMU detects upward tilt
    4. Push forward until level again
    """

    print("Climbing step...")

    # push wheels up strongly
    left_motor.setVelocity(6.0)
    right_motor.setVelocity(6.0)

    # wait until the robot tilts upward
    while robot.step(timestep) != -1:
        roll, pitch, yaw = imu.getRollPitchYaw()
        if pitch < -0.2:  # climbing angle detected
            break

    # continue pushing until robot becomes level again
    while robot.step(timestep) != -1:
        roll, pitch, yaw = imu.getRollPitchYaw()
        if abs(pitch) < 0.05:  # plateau reached
            break

    stop()
    robot.step(200)  # small delay

# ------------------------------------------------
# Main Loop
# ------------------------------------------------

print("✅ Stair-Climbing Cleaning Robot Started")

while robot.step(timestep) != -1:

    f = front_sensor.getValue()
    b = bottom_sensor.getValue()
    roll, pitch, yaw = imu.getRollPitchYaw()

    # NORMAL CLEANING MODE
    if b < 200:  
        # safe floor area
        forward(4.0)
    
    # DETECT EDGE (stairs going down) → avoid falling
    if b > 600:
        print("⚠️ Edge detected — Turning back")
        stop()
        robot.step(200)
        turn_left()
        robot.step(500)
        continue

    # DETECT STAIR UPWARD STEP
    if f > 700:
        print("🟦 Stair step detected — Preparing to climb")
        stop()
        robot.step(300)
        climb_step()

    # SAFETY: if tilt too large, stop
    if abs(pitch) > 0.5:
        print("❗Unstable tilt — stopping")
        stop()
        continue
