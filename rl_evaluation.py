import time

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from dwa_obstacles_env import DwaEnv


def report_time(name, data):
    data = np.asarray(data, dtype=float)
    mean_time = np.mean(data)
    row = {
        "name": name,
        "mean_ms": mean_time * 1000.0,
        "std_ms": np.std(data) * 1000.0,
        "max_ms": np.max(data) * 1000.0,
        "p95_ms": np.percentile(data, 95) * 1000.0,
        "frequency_hz": 1.0 / mean_time,
    }

    print(f"\n{name}")
    print(f"Mean time: {row['mean_ms']:.3f} ms")
    print(f"Std time:  {row['std_ms']:.3f} ms")
    print(f"Max time:  {row['max_ms']:.3f} ms")
    print(f"95% time:  {row['p95_ms']:.3f} ms")
    print(f"Frequency: {row['frequency_hz']:.2f} Hz")
    return row


def main():
    env = DwaEnv(verbose=False)
    model = SAC.load("sac_dwa_model_3.zip", env=env)

    success_count = 0
    collision_count = 0
    timeout_count = 0
    speedlow_count = 0
    step_total_success = 0
    step_total_all = 0

    sac_times = []
    planning_times = []
    cycle_times = []

    for i in range(400):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        info = {}

        while not done:
            t0 = time.perf_counter()

            t_sac_0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            t_sac_1 = time.perf_counter()

            t_plan_0 = time.perf_counter()
            obs, reward, done, info = env.step(action)
            t_plan_1 = time.perf_counter()

            t1 = time.perf_counter()

            sac_times.append(t_sac_1 - t_sac_0)
            planning_times.append(t_plan_1 - t_plan_0)
            cycle_times.append(t1 - t0)

            total_reward += reward
            step += 1
            step_total_all += 1

        result = info.get("result", "")
        if result == "success":
            success_count += 1
            step_total_success += step
        elif result == "collision":
            collision_count += 1
        elif result == "timeout":
            timeout_count += 1
        elif result == "speedlow":
            speedlow_count += 1

        print(
            f"Episode {i + 1}: total reward = {total_reward:.2f}, "
            f"result = {result}, steps = {step}"
        )

    if success_count > 0:
        print(f"Average successful steps = {step_total_success / success_count:.2f}")
    else:
        print("Average successful steps = N/A (no successful episodes)")

    print(f"\nSuccess count: {success_count}")
    print(f"Collision count: {collision_count}")
    print(f"Timeout count: {timeout_count}")
    print(f"Speed-low count: {speedlow_count}")

    timing_rows = [
        report_time("SAC inference", sac_times),
        report_time("CBF-DWA planning and state update", planning_times),
        report_time("Complete control cycle", cycle_times),
    ]

    pd.DataFrame(timing_rows).to_excel("real_time_evaluation.xlsx", index=False)
    print("\nTiming statistics saved to: real_time_evaluation.xlsx")
    print(f"Total measured control steps: {step_total_all}")
    print(f"Required control period: {env.dwa.dt * 1000.0:.1f} ms")
    print(f"Required minimum frequency: {1.0 / env.dwa.dt:.2f} Hz")


if __name__ == "__main__":
    main()
