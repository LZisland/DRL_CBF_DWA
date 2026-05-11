import math
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from multi_obstacle_cbf import DWA
from quadrotormodel import KinematicModel


def make_sphere(center, radius, n=24):
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_sphere(ax, center, radius, color="tab:blue", alpha=0.14):
    x, y, z = make_sphere(center, radius)
    ax.plot_surface(x, y, z, alpha=alpha, linewidth=0.0, color=color)


def draw_box_from_segment(ax, start, end, half_width, half_height, color="black"):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    center = 0.5 * (start + end)
    length = np.linalg.norm(end - start)

    if abs(start[0] - end[0]) < 1e-9:
        size = np.array([2.0 * half_width, length, 2.0 * half_height], dtype=float)
    else:
        size = np.array([length, 2.0 * half_width, 2.0 * half_height], dtype=float)

    cx, cy, cz = center
    sx, sy, sz = size / 2.0
    corners = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linewidth=1.0,
        )


def generate_variable_radius_spheres_for_segment(
    start,
    end,
    half_width,
    half_height,
    num_spheres=4,
    safety_factor=1.03,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    axis_vec = end - start
    length = np.linalg.norm(axis_vec)

    if length < 1e-12:
        rho = np.sqrt(half_width ** 2 + half_height ** 2)
        return [(start, safety_factor * rho)]

    direction = axis_vec / length
    rho = np.sqrt(half_width ** 2 + half_height ** 2)
    num_spheres = max(2, int(num_spheres))
    s_list = np.linspace(0.0, length, num_spheres)

    spheres = []
    for i, s_i in enumerate(s_list):
        center = start + direction * s_i
        if i == 0:
            a_i = 0.0
            b_i = 0.5 * (s_list[i] + s_list[i + 1])
        elif i == num_spheres - 1:
            a_i = 0.5 * (s_list[i - 1] + s_list[i])
            b_i = length
        else:
            a_i = 0.5 * (s_list[i - 1] + s_list[i])
            b_i = 0.5 * (s_list[i] + s_list[i + 1])

        axial_span = max(s_i - a_i, b_i - s_i)
        radius = safety_factor * np.sqrt(rho ** 2 + axial_span ** 2)
        spheres.append((center, radius))
    return spheres


def build_elongated_obstacles():
    half_width = 0.23
    half_height = 0.20
    segments = [
        (np.array([2.8, 4.0, 4.0]), np.array([5.2, 4.0, 4.0])),
        (np.array([5.8, 7.0, 7.2]), np.array([8.2, 7.0, 7.2])),
        (np.array([7.2, 8.7, 8.8]), np.array([9.4, 8.7, 8.8])),
    ]

    spheres = []
    for start, end in segments:
        spheres.extend(
            generate_variable_radius_spheres_for_segment(
                start,
                end,
                half_width=half_width,
                half_height=half_height,
                num_spheres=3,
                safety_factor=1.03,
            )
        )

    obstacle_array = np.array([[c[0], c[1], c[2], r] for c, r in spheres], dtype=float)
    return segments, obstacle_array


def envelope_surface_distances(state, obstacles):
    centers = obstacles[:, :3]
    radii = obstacles[:, 3]
    return np.linalg.norm(state[:3] - centers, axis=1) - radii


def make_distance_record(time_value, state, obstacles, quadrotor_radius):
    distances = envelope_surface_distances(state, obstacles)
    record = {"time": time_value}
    for idx, distance in enumerate(distances, start=1):
        record[f"sphere_{idx}_distance"] = distance
    record["min_envelope_surface_distance"] = float(np.min(distances))
    record["min_clearance_to_safe_boundary"] = float(np.min(distances) - quadrotor_radius)
    return record


def save_distance_history(distance_history, output_path):
    df = pd.DataFrame(distance_history)
    df.to_excel(output_path, index=False)


def save_trajectory_history(trajectory, dt, output_path):
    trajectory = np.asarray(trajectory, dtype=float)
    df = pd.DataFrame(
        trajectory,
        columns=[
            "position_x",
            "position_y",
            "position_z",
            "velocity_xoy",
            "velocity_z",
            "velocity_w",
            "yaw",
        ],
    )
    df.insert(0, "time", np.arange(len(df)) * dt)
    df.to_excel(output_path, index=False)


class EnvelopeDWA(DWA):
    def trajectory_predict(self, state_init, vxoy, vz, vw):
        state = np.array(state_init, dtype=float).copy()
        trajectory = [state.copy()]
        time = 0.0

        while time <= self.predict_time:
            state = KinematicModel(state.copy(), [vxoy, vz, vw], self.dt)
            trajectory.append(state.copy())
            time += self.dt

        return np.asarray(trajectory)

    def dist(self, state, obstacle):
        obstacle = np.asarray(obstacle, dtype=float)
        centers = obstacle[:, :3]
        radii = obstacle[:, 3]
        d = np.linalg.norm(state[:3] - centers, axis=1) - (radii + self.quadrotor_radius)
        return float(np.min(d))

    def cbf_simple(self, trajectory, obstacle):
        mu = 0.1
        cbf_min = float("inf")
        obstacle = np.asarray(obstacle, dtype=float)

        for obs in obstacle:
            obs_pos = obs[:3]
            safe_distance = obs[3] + self.quadrotor_radius
            for i in range(len(trajectory) - 1):
                pos = trajectory[i, 0:3]
                yaw = trajectory[i, 6]
                vxy = trajectory[i, 3]
                vz = trajectory[i, 4]
                vel_vec = np.array(
                    [vxy * math.cos(yaw), vxy * math.sin(yaw), vz],
                    dtype=float,
                )

                delta_p = pos - obs_pos
                h_k = np.dot(delta_p, delta_p) - safe_distance ** 2
                next_pos = pos + self.dt * vel_vec
                delta_p_next = next_pos - obs_pos
                h_k1 = np.dot(delta_p_next, delta_p_next) - safe_distance ** 2
                cbf_val = (h_k1 - h_k) + self.delta * h_k
                cbf_min = min(cbf_min, cbf_val)

        return cbf_min


def simulate():
    planner = EnvelopeDWA()
    planner.alpha = 8.0
    planner.beta = 4.0
    planner.gamma = 0.1
    planner.target = np.array([11.0, 11.0, 11.0], dtype=float)

    segments, obstacles = build_elongated_obstacles()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    trajectory = [state.copy()]
    distance_history = [
        make_distance_record(0.0, state, obstacles, planner.quadrotor_radius)
    ]
    min_margin = float("inf")
    slow_steps = 0
    result = "timeout"

    for step in range(500):
        control, predicted_trajectory, _, _, _ = planner.dwa_control(state, planner.target, obstacles)
        state = KinematicModel(state.copy(), control, planner.dt)
        trajectory.append(state.copy())
        distance_history.append(
            make_distance_record((step + 1) * planner.dt, state, obstacles, planner.quadrotor_radius)
        )

        margin = planner.dist(state, obstacles)
        min_margin = min(min_margin, margin)

        if np.linalg.norm(state[:3] - planner.target) <= planner.quadrotor_radius:
            result = "success"
            break

        speed = math.hypot(state[3], state[4])
        if speed < 0.1:
            slow_steps += 1
        else:
            slow_steps = 0

        if margin <= 0.0:
            result = "collision"
            break

        if slow_steps > 100:
            result = "speedlow"
            break
    else:
        step = 499

    trajectory = np.asarray(trajectory)
    avg_speed = float(np.mean(np.hypot(trajectory[:, 3], trajectory[:, 4])))
    path_length = float(np.sum(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1)))

    print("Scenario: elongated obstacles in the long-range CBF-DWA environment")
    print(f"Result: {result}")
    print(f"Steps: {step + 1}")
    print(f"Minimum body-to-obstacle margin: {min_margin:.4f} m")
    print(f"Average speed: {avg_speed:.4f} m/s")
    print(f"Path length: {path_length:.4f} m")
    print(f"Envelope spheres: {len(obstacles)}")
    save_distance_history(
        distance_history,
        "elongated_obstacle_distances.xlsx",
    )
    save_trajectory_history(
        trajectory,
        planner.dt,
        "elongated_uav_trajectory.xlsx",
    )
    print("Distance data saved to: elongated_obstacle_distances.xlsx")
    print("Trajectory data saved to: elongated_uav_trajectory.xlsx")


if __name__ == "__main__":
    simulate()
