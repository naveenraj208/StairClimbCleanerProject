# planner_utils.py
# Utilities for occupancy grid, A* planner, raycasting and map IO

import heapq
import math
import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None

def world_to_map(x, y, origin, resolution):
    """
    Convert world coords (x, z) to map indices (row, col).
    origin: (ox, oy) world coordinate of map (lower-left)
    resolution: meters per cell
    using x -> col, y -> row (y is forward/z in Webots)
    """
    ox, oy = origin
    col = int((x - ox) / resolution)
    row = int((y - oy) / resolution)
    return row, col

def map_to_world(row, col, origin, resolution):
    ox, oy = origin
    x = ox + (col + 0.5) * resolution
    y = oy + (row + 0.5) * resolution
    return x, y

def raycast_grid(origin_xy, theta, ranges, max_range, grid, origin, resolution):
    """
    Insert lidar range readings into occupancy grid using simple Bresenham / ray march.
    origin_xy: robot (x,y) world position
    theta: heading (radians)
    ranges: list/array of range measurements (in meters)
    grid: numpy array (rows, cols) float occupancy probability 0..1
    origin: world origin of grid lower-left
    resolution: meters per cell
    Assumes lidar angles are uniformly across FOV and data order corresponds to angles.
    """
    rows, cols = grid.shape
    ox, oy = origin
    n = len(ranges)
    for i, r in enumerate(ranges):
        if not np.isfinite(r) or r <= 0.0:
            continue
        # angle of this beam relative to robot heading (assume symmetric FOV across ranges length)
        ang = - (n - 1) / 2.0 + i  # index-centered
        # Convert index to angle using FOV normalization is caller's job; here we assume caller passes pre-rotated angles
        beam_angle = theta + ang
        # end point world coords (x_world corresponds to x, y_world to z)
        ex = origin_xy[0] + r * math.cos(beam_angle)
        ey = origin_xy[1] + r * math.sin(beam_angle)
        # rasterize line from origin to endpoint
        mark_free_line(grid, origin_xy, (ex, ey), origin, resolution)
        # mark endpoint occupied
        rr, cc = world_to_map(ex, ey, origin, resolution)
        if 0 <= rr < rows and 0 <= cc < cols:
            grid[rr, cc] = min(1.0, grid[rr, cc] + 0.7)  # high prob occupied

def mark_free_line(grid, p0, p1, origin, resolution):
    """
    Bresenham-ish sampling along the segment p0->p1 marking cells as free (decreasing occupancy).
    """
    rows, cols = grid.shape
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = int(max(1, (dist / resolution) * 2))
    for s in range(steps):
        t = s / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        rr, cc = world_to_map(x, y, origin, resolution)
        if 0 <= rr < rows and 0 <= cc < cols:
            grid[rr, cc] = max(0.0, grid[rr, cc] - 0.2)  # lower occupancy

# -----------------------------
# A* planner
# -----------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal, obstacle_threshold=0.5):
    """
    grid: 2D numpy array with occupancy prob [0..1] (higher = more blocked)
    start, goal: (row, col)
    returns: list of (row,col) path or None
    """
    rows, cols = grid.shape
    def neighbors(node):
        r, c = node
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] < obstacle_threshold:
                    yield (nr, nc)
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start,goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    while open_set:
        f, cost, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            # reconstruct path
            path = [current]
            p = parent
            while p:
                path.append(p)
                p = came_from.get(p)
            path.reverse()
            return path
        for nb in neighbors(current):
            tentative_g = gscore[current] + math.hypot(nb[0]-current[0], nb[1]-current[1])
            if tentative_g < gscore.get(nb, float('inf')):
                gscore[nb] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(nb,goal), tentative_g, nb, current))
    return None

# -----------------------------
# Save map to image
# -----------------------------
def save_map_png(grid, filename):
    if Image is None:
        return
    # grid values 0..1 => 255..0 (free to occupied)
    arr = (255 * (1.0 - np.clip(grid, 0.0, 1.0))).astype(np.uint8)
    im = Image.fromarray(arr)
    im = im.convert("L")
    im.save(filename)
