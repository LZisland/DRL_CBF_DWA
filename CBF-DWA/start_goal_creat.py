import random
import math
import pandas as pd
from typing import List, Tuple

# --------- 参数配置 ---------
MAP_SIZE = 8.0  # 环境边长
OBSTACLES: List[Tuple[float, float, float]] = [
    (3.0, 3.0, 5.0),
    (5.0, 2.0, 2.0),
    (2.0, 5.0, 3.0),
]
OBSTACLE_RADIUS = 1.0  # 障碍物半径
CLEARANCE = 0.0        # 起终点与障碍物最小间隙
MIN_DISTANCE = 7.0     # 起点到终点最小距离
N_PAIRS = 4000         # 需要采样的对数

Point = Tuple[float, float, float]

def distance(p: Point, q: Point) -> float:
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2)

def sample_point(map_size: float,
                 obstacles: List[Point],
                 obstacle_radius: float,
                 clearance: float) -> Point:
    while True:
        p = (random.uniform(0, map_size),
             random.uniform(0, map_size),
             random.uniform(0, map_size))
        if all(distance(p, c) >= obstacle_radius + clearance for c in obstacles):
            return p

def segment_intersects_sphere(p1: Point,
                              p2: Point,
                              center: Point,
                              radius: float) -> bool:
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    fx, fy, fz = p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]
    a = dx*dx + dy*dy + dz*dz
    b = 2 * (fx*dx + fy*dy + fz*dz)
    c = fx*fx + fy*fy + fz*fz - radius*radius
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def path_through_obstacle(p1: Point,
                          p2: Point,
                          obstacles: List[Point],
                          radius: float) -> bool:
    return any(segment_intersects_sphere(p1, p2, c, radius) for c in obstacles)

# 生成点对
starts = []
goals = []
count = 0
while count < N_PAIRS:
    start = sample_point(MAP_SIZE, OBSTACLES, OBSTACLE_RADIUS, CLEARANCE)
    while True:
        goal = sample_point(MAP_SIZE, OBSTACLES, OBSTACLE_RADIUS, CLEARANCE)
        if distance(start, goal) >= MIN_DISTANCE:
            break
    if path_through_obstacle(start, goal, OBSTACLES, OBSTACLE_RADIUS):
        starts.append(start)
        goals.append(goal)
        count += 1

# 保存到 Excel
df_pairs = pd.DataFrame({
    'start_x': [s[0] for s in starts],
    'start_y': [s[1] for s in starts],
    'start_z': [s[2] for s in starts],
    'goal_x': [g[0] for g in goals],
    'goal_y': [g[1] for g in goals],
    'goal_z': [g[2] for g in goals],
})
excel_path = 'start_goal_pairs_100.xlsx'
df_pairs.to_excel(excel_path, index=False)
