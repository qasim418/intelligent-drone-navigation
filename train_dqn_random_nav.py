


"""DQN training for point-to-point drone navigation.

This script trains a discrete-action DQN agent in a PyBullet environment.
Each episode samples a random start position and a random goal position
within a circular arena populated with obstacles.
"""

import math
import os
from datetime import datetime
import time
from pathlib import Path
from typing import Tuple

from packaging import version

import numpy as np
import pybullet as p
from stable_baselines3 import DQN, __version__ as SB3_VERSION
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from forest_navigation_env import EnhancedForestWithObstacles


# --------------------------- Training configuration ------------------------------------
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 2e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 500
ACTION_REPEAT = 4
ARENA_RADIUS = 50.0  # 100 x 100 meters
GOAL_MIN_DIST = 3.0
GOAL_MAX_DIST = 5.0
SUCCESS_THRESHOLD = 0.5
PROXIMITY_THRESHOLD = 1.0
PROXIMITY_PENALTY = -0.5
COLLISION_PENALTY = -15.0
SUCCESS_REWARD = 20.0
# Where to store training artifacts (matches existing workspace structure).
RUNS_ROOT = Path(__file__).resolve().parent / "training_sessions"
CHECKPOINT_MILESTONES = [500_000]
EVAL_FREQUENCY = 50_000
EVAL_EPISODES = 5

import gymnasium as gym
from gymnasium import spaces


class TrainingLogger(BaseCallback):
    """Records basic training throughput (FPS) for observability."""

    def __init__(self, check_freq: int = 5000):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = max(time.time() - self.start_time, 1e-6)
            fps = self.num_timesteps / elapsed
            self.logger.record("custom/fps", fps)
        return True


class MilestoneCheckpoint(BaseCallback):
    """Saves model checkpoints when reaching specified timesteps."""

    def __init__(self, milestones, save_dir, prefix="dqn_random_point_nav"):
        super().__init__()
        self.milestones = sorted(milestones)
        self.save_dir = save_dir
        self.prefix = prefix
        self.saved = set()
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for milestone in self.milestones:
            if milestone in self.saved:
                continue
            if self.num_timesteps >= milestone:
                path = os.path.join(self.save_dir, f"{self.prefix}_{milestone // 1000}k")
                self.model.save(path)
                print(f"[Checkpoint] Saved model at {milestone:,} steps -> {path}.zip")
                self.saved.add(milestone)
        return True


class RandomPointNavEnv(gym.Env):
    """Gymnasium environment for drone navigation with random start/goal pairs."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = False):
        super().__init__()
        self.gui = gui

        # Create the PyBullet scene (procedural forest + static obstacles).
        self.scene = EnhancedForestWithObstacles(gui=gui, radius=ARENA_RADIUS)
        self.client = self.scene.client
        self.drone_id = self.scene.drone_id
        self.radius = self.scene.radius

        # Visual marker for the goal location.
        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.4, rgbaColor=[1.0, 0.1, 0.1, 0.8], physicsClientId=self.client
        )
        self.goal_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_vis, basePosition=[0, 0, 1.0], physicsClientId=self.client
        )

        # Discrete action space: hover, then translations along ±X/±Y/±Z.
        self.action_space = spaces.Discrete(7)
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            2: np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            4: np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32),
            5: np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            6: np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32),
        }
        self.linear_speed = 3.0

        # Observation space: 36-beam LiDAR + normalized goal distance + relative goal angle.
        # LiDAR uses a short range to focus on immediate obstacle avoidance.
        self.lidar_rays = 36
        self.lidar_range = 2.0  # short range for immediate obstacle detection
        low = np.array([0.0] * self.lidar_rays + [0.0, -np.pi], dtype=np.float32)
        high = np.array([1.0] * self.lidar_rays + [1.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.prev_dist = None
        self.start_pos = None
        self._rng = np.random.default_rng()
        # NOTE: keep `start_pos` as a public attribute; other files may read it.

    # ------------------------------------------------------------------ Helpers
    def _sample_valid_point(self, margin: float = 2.0) -> Tuple[float, float]:
        for _ in range(100):
            r = self._rng.uniform(margin, self.radius - margin)
            ang = self._rng.uniform(0, 2 * math.pi)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            if self.scene._valid(x, y, min_sep=0.5):
                return x, y
        return 0.0, 0.0

    def _sample_goal(self, start_xy: np.ndarray) -> np.ndarray:
        for _ in range(100):
            dist = self._rng.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
            ang = self._rng.uniform(0, 2 * math.pi)
            gx = start_xy[0] + dist * math.cos(ang)
            gy = start_xy[1] + dist * math.sin(ang)
            if math.hypot(gx, gy) <= self.radius - 1.0 and self.scene._valid(gx, gy, min_sep=0.5):
                return np.array([gx, gy, 1.0], dtype=np.float32)
        return np.array([start_xy[0], start_xy[1], 1.0], dtype=np.float32)

    # ---------------------------------------------------------------- Utilities
    def _get_lidar(self, position, yaw_angle):
        ray_origins, ray_targets = [], []
        # Full 360-degree scan around the drone.
        for i in range(self.lidar_rays):
            scan_angle = yaw_angle + i * (2 * math.pi / self.lidar_rays)
            ray_origins.append([position[0], position[1], position[2]])
            ray_targets.append(
                [
                    position[0] + self.lidar_range * math.cos(scan_angle),
                    position[1] + self.lidar_range * math.sin(scan_angle),
                    position[2],
                ]
            )

        results = p.rayTestBatch(ray_origins, ray_targets, physicsClientId=self.client)
        lidar_hits = []
        for res in results:
            hit_fraction = res[2]
            lidar_hits.append(1.0 if hit_fraction < 0 else hit_fraction)
        return np.array(lidar_hits, dtype=np.float32)

    def _proximity_penalty(self, lidar_hits: np.ndarray) -> float:
        min_norm = float(np.min(lidar_hits))
        min_dist = min_norm * self.lidar_range
        if min_dist < PROXIMITY_THRESHOLD:
            ratio = (PROXIMITY_THRESHOLD - min_dist) / PROXIMITY_THRESHOLD
            return PROXIMITY_PENALTY * ratio
        return 0.0

    def _get_obs(self):
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        lidar_hits = self._get_lidar(position, yaw)

        dist = np.linalg.norm(self.goal_pos[:2] - np.array(position[:2]))
        rel_angle = (
            math.atan2(self.goal_pos[1] - position[1], self.goal_pos[0] - position[0]) - yaw + math.pi
        ) % (2 * math.pi) - math.pi
        goal_dist_norm = min(dist / GOAL_MAX_DIST, 1.0)
        return np.concatenate([lidar_hits, np.array([goal_dist_norm, rel_angle], dtype=np.float32)])

    # ---------------------------------------------------------------------- Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(int(self.np_random.integers(1 << 63)))

        start_x, start_y = self._sample_valid_point()
        start_position = np.array([start_x, start_y, 1.0], dtype=np.float32)
        self.start_pos = start_position.copy()
        self.goal_pos = self._sample_goal(start_position[:2])

        p.resetBasePositionAndOrientation(
            self.drone_id, start_position.tolist(), [0, 0, 0, 1], physicsClientId=self.client
        )
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client)

        self.prev_dist = np.linalg.norm(self.goal_pos[:2] - start_position[:2])
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        action_vector = self.action_map[int(action)]
        delta_x, delta_y, delta_z, throttle = action_vector

        _, current_orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(current_orn)[2]

        vel_x = (delta_x * math.cos(yaw) - delta_y * math.sin(yaw)) * self.linear_speed * throttle
        vel_y = (delta_x * math.sin(yaw) + delta_y * math.cos(yaw)) * self.linear_speed * throttle
        vel_z = delta_z * self.linear_speed * throttle

        p.resetBaseVelocity(self.drone_id, [vel_x, vel_y, vel_z], [0, 0, 0], physicsClientId=self.client)
        for _ in range(ACTION_REPEAT):
            p.stepSimulation(physicsClientId=self.client)

        next_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        next_pos = np.array(next_pos)
        distance_to_goal = np.linalg.norm(self.goal_pos[:2] - next_pos[:2])

        lidar = self._get_lidar(next_pos, yaw)
        reward = 0.0

        # Reward shaping: positive when moving closer to the goal.
        if self.prev_dist is not None:
            reward += self.prev_dist - distance_to_goal
        self.prev_dist = distance_to_goal

        # Penalty shaping: negative when too close to obstacles.
        reward += self._proximity_penalty(lidar)

        terminated = False
        truncated = False

        # Terminal event: collision with any obstacle.
        if p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client):
            reward += COLLISION_PENALTY
            terminated = True

        # Terminal event: reaching the goal region.
        if distance_to_goal < SUCCESS_THRESHOLD:
            reward += SUCCESS_REWARD
            terminated = True

        obs = np.concatenate(
            [
                lidar,
                np.array(
                    [
                        min(distance_to_goal / GOAL_MAX_DIST, 1.0),
                        (
                            (
                                math.atan2(self.goal_pos[1] - next_pos[1], self.goal_pos[0] - next_pos[0])
                                - yaw
                                + math.pi
                            )
                            % (2 * math.pi)
                            - math.pi
                        ),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass


def run_training():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / timestamp
    snapshots_dir = run_dir / "snapshots"
    tensorboard_dir = run_dir / "tensorboard"
    evaluations_dir = run_dir / "evaluations"
    metrics_dir = run_dir / "metrics"

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(RandomPointNavEnv(gui=False))
    eval_env = Monitor(RandomPointNavEnv(gui=False))

    checkpoint_cb = MilestoneCheckpoint(CHECKPOINT_MILESTONES, str(snapshots_dir))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(snapshots_dir),
        log_path=str(evaluations_dir),
        eval_freq=EVAL_FREQUENCY,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    callback = CallbackList([TrainingLogger(), checkpoint_cb, eval_cb])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device="auto",
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    finally:
        model_path = run_dir / f"final_model_{timestamp}"
        model.save(str(model_path))
        print(f"Saved model to {model_path}.zip")

        try:
            summary_path = run_dir / "training_summary.txt"
            summary_path.write_text(
                "\n".join(
                    [
                        f"timestamp: {timestamp}",
                        f"sb3_version: {SB3_VERSION}",
                        f"total_timesteps: {TOTAL_TIMESTEPS}",
                        f"final_model: {model_path}.zip",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        env.close()
        eval_env.close()


def train():
    """Backward-compatible entrypoint."""
    return run_training()


if __name__ == "__main__":
    run_training()



