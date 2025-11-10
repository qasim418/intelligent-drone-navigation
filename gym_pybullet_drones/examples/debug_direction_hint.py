"""Utility script to inspect target sampling and direction hints in PointToPointAviary."""

import argparse
from typing import Optional

import numpy as np

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.enums import ObservationType
from gym_pybullet_drones.utils.utils import str2bool as utils_str2bool


def main(count: int, seed: Optional[int], obs_type: str, randomize_start: bool, randomize_target: bool) -> None:
    obs_type = obs_type.lower()
    if obs_type not in {"kin", "rgb"}:
        raise ValueError("obs_type must be 'kin' or 'rgb'")
    obs_enum = ObservationType.RGB if obs_type == "rgb" else ObservationType.KIN

    env = PointToPointAviary(
        gui=False,
        randomize_start=randomize_start,
        randomize_target=randomize_target,
        use_built_in_obstacles=False,
        ctrl_freq=24 if obs_enum == ObservationType.RGB else 30,
        obs=obs_enum,
        success_snapshot_dir=None,
        seed=seed,
    )

    print("Sampled episodes:")
    for episode in range(count):
        obs, info = env.reset()
        start = env._start_position.copy()
        goal = env._target_position.copy()
        hint = info.get("direction_hint")
        hint_index = info.get("direction_hint_index")
        print(f"Episode {episode+1:02d}")
        print(f"  Start position : {start}")
        print(f"  Target position: {goal}")
        print(f"  Direction hint : {hint} (index {hint_index})")
        print(f"  Distance       : {np.linalg.norm(goal - start):.3f} m")
        if episode != count - 1:
            print("-")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect direction hints emitted by PointToPointAviary")
    parser.add_argument("--episodes", type=int, default=5, help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--obs_type", type=str, default="rgb", choices=["kin", "rgb"],
                        help="Observation type for the environment")
    parser.add_argument("--randomize_start", type=utils_str2bool, default=True,
                        help="Randomize the start position when not fixed")
    parser.add_argument("--randomize_target", type=utils_str2bool, default=True,
                        help="Randomize the goal position when not fixed")
    args = parser.parse_args()

    main(
        count=args.episodes,
        seed=args.seed,
        obs_type=args.obs_type,
        randomize_start=args.randomize_start,
        randomize_target=args.randomize_target,
    )
