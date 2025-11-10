"""Utility to capture sample RGB observations from PointToPointAviary."""

import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image

from gym_pybullet_drones.envs.PointToPointAviary import PointToPointAviary
from gym_pybullet_drones.utils.enums import ObservationType


def capture_samples(
    output_dir: str,
    num_samples: int,
    gui: bool,
    seed: Optional[int],
    use_built_in_obstacles: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    env = PointToPointAviary(
        gui=gui,
        randomize_start=True,
        randomize_target=True,
        velocity_scale=1.5,
        obs=ObservationType.RGB,
        ctrl_freq=24,
        use_built_in_obstacles=use_built_in_obstacles,
        success_snapshot_dir=None,
        seed=seed,
    )

    try:
        obs, info = env.reset(seed=seed)
        step_count = 0
        saved = 0

        while saved < num_samples:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)

            rgb_full = obs[0]
            print(
                f"[debug] obs dtype={rgb_full.dtype}, shape={rgb_full.shape}, min={rgb_full.min()}, max={rgb_full.max()}, mean={rgb_full.mean():.4f}"
            )
            channels_stats = [
                (
                    idx,
                    int(rgb_full[:, :, idx].min()),
                    int(rgb_full[:, :, idx].max()),
                    float(rgb_full[:, :, idx].mean()),
                )
                for idx in range(rgb_full.shape[2])
            ]
            print("[debug] per-channel stats (index, min, max, mean):")
            for idx, cmin, cmax, cmean in channels_stats:
                print(f"  ch{idx}: min={cmin} max={cmax} mean={cmean:.2f}")
            rgb = np.array(rgb_full[:, :, :3], copy=True)
            if not isinstance(rgb, np.ndarray):
                rgb = np.asarray(rgb)

            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating):
                    max_val = float(rgb.max()) if rgb.size > 0 else 0.0
                    min_val = float(rgb.min()) if rgb.size > 0 else 0.0
                    print(f"[debug] frame {saved}: float image stats pre-scale min={min_val:.3f} max={max_val:.3f}")
                    if max_val <= 1.0:
                        rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]

            print(
                f"[debug] frame {saved}: uint8 stats min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.2f}, shape={rgb.shape}"
            )

            filename = f"sample_{saved:03d}_step{step_count:05d}.png"
            filepath = os.path.join(output_dir, filename)
            Image.fromarray(rgb).save(filepath)
            print(f"[info] saved sample to {os.path.abspath(filepath)}")
            saved += 1

            if terminated or truncated:
                obs, info = env.reset()
                step_count = 0
            else:
                step_count += 1

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture sample RGB frames from the P2P aviary.")
    parser.add_argument("output", type=str, help="Directory where sample images will be stored.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of frames to capture (default: 10).")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI while capturing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no_built_in_obstacles",
        action="store_true",
        help="Disable built-in obstacles when capturing (enabled by default).",
    )

    args = parser.parse_args()
    capture_samples(
        output_dir=args.output,
        num_samples=args.num_samples,
        gui=args.gui,
        seed=args.seed,
        use_built_in_obstacles=not args.no_built_in_obstacles,
    )


if __name__ == "__main__":
    main()
