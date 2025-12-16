#!/usr/bin/env python3
"""
enhanced_forest_with_obstacles.py

Procedural PyBullet arena: forest clutter + a few explicit obstacles/buildings.

Usage:
    python enhanced_forest_with_obstacles.py

Exit:
    Press ESC or 'q' in the PyBullet window (or just close the window).

Quick tweaks:
    - Adjust SEED/NUM_* values to change density and layout.
    - ENABLE_SHADOWS may help (or hurt) depending on your GPU.
"""

import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import pkg_resources
import os

# Optional OpenCV support for showing the drone camera feed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Drone camera view will be disabled.")

# ---------------- PARAMETERS ----------------
SEED = 1234
GUI_FPS = 240.0
ARENA_RADIUS = 50.0  # 100x100 playable area (radius 50)
SIMULATION_SPEED = 2.0  # Simulation speed multiplier (0.1 = 10x slower, 1.0 = real-time, 2.0 = 2x faster)

NUM_TREES = 140
NUM_BUSHES = 70
NUM_GRASS = 1
NUM_ROCKS = 24
NUM_LOGS = 10
NUM_MUSHROOM_CLUSTERS = 1
LEAF_CLUMPS = 1
NUM_OBSTACLES = 10  # Number of explicit navigation obstacles
NUM_BUILDINGS = 3   # Number of buildings

NUM_HUMANS = 2
HUMAN_HEIGHT = 1.6
HUMAN_RADIUS = 0.2
# Human marker color (choose something that stands out in the scene)
HUMAN_COLOR = [1.0, 0.0, 1.0, 1.0]  # Bright magenta

CLEARING = True
CLEARING_RADIUS = 2.0  # keep a small clearing around center (optional)

KEEP_GUI = True   # True -> p.GUI , False -> p.DIRECT
ENABLE_SHADOWS = False  # Toggle shadows for your GPU
# ----------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def rand_in_disk(radius):
    """Sample (x, y) uniformly from a disk with the given radius."""
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    return r * math.cos(theta), r * math.sin(theta)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EnhancedForestWithObstacles:
    def __init__(self,
                 gui=KEEP_GUI,
                 radius=ARENA_RADIUS,
                 n_trees=NUM_TREES,
                 n_bushes=NUM_BUSHES,
                 n_grass=NUM_GRASS,
                 n_rocks=NUM_ROCKS,
                 n_logs=NUM_LOGS,
                 n_mushrooms=NUM_MUSHROOM_CLUSTERS,
                 leaf_clumps=LEAF_CLUMPS,
                 n_obstacles=NUM_OBSTACLES,
                 n_buildings=NUM_BUILDINGS,
                 clearing=CLEARING,
                 clearing_radius=CLEARING_RADIUS):
        self.gui = gui
        self.radius = radius
        self.n_trees = n_trees
        self.n_bushes = n_bushes
        self.n_grass = n_grass
        self.n_rocks = n_rocks
        self.n_logs = n_logs
        self.n_mushrooms = n_mushrooms
        self.leaf_clumps = leaf_clumps
        self.n_obstacles = n_obstacles
        self.n_buildings = n_buildings
        self.clearing = clearing
        self.clearing_radius = clearing_radius

        # Human placement / labels
        self.num_humans = NUM_HUMANS
        self.human_ids = []
        self.human_positions = []
        self.human_label_ids = []
        self.human_target_direction = None  # Will be set later

        # Bookkeeping for simple overlap avoidance during placement
        self.placed = []

        # Connect to PyBullet
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # Basic visualizer settings
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if ENABLE_SHADOWS else 0,
                                   physicsClientId=self.client)

        # Assemble the scene
        self._setup_ground()
        self._place_obstacles()         # explicit navigation blockers (pillars/walls/crates)
        self._place_buildings()         # large static structures
        self._place_trees()
        self._place_bushes_and_grass()
        self._place_rocks()
        self._place_logs()
        self._place_mushrooms()
        self._place_leaf_clumps()
        self._place_center_marker()
        self._add_boundary_direction_labels()

        # Add humans after the static world so labels are visible
        self._place_humans()

        # Add a visible drone (URDF if available, otherwise a placeholder)
        self._load_drone()
        
        # Follow-camera configuration (arrow keys adjust yaw/pitch, +/- zoom)
        # Keep the default closer in GUI so the drone is easier to see.
        self.camera_distance = 3.0 if self.gui else 6.0
        self.camera_pitch = -35.0
        self.camera_yaw_offset = 0.0
        self._camera_yaw = self.camera_yaw_offset
        self._camera_pitch = self.camera_pitch
        self.show_lidar_overlay = False

        
        # Drone visual scale (placeholder drone uses this)
        self.drone_scale = 2.0  # Scale factor for drone size
        
        # Drone camera feed (requires OpenCV)
        self.show_drone_camera = True  # Enable/disable drone camera view
        self.drone_camera_width = 480  # Camera image width
        self.drone_camera_height = 360  # Camera image height
        self.drone_camera_fov = 80  # Field of view (degrees)
        self.drone_camera_update_freq = 2  # Update camera every N frames (for performance)
        self.drone_camera_frame_count = 0
        
        # Display mode: 'window' (separate OpenCV window), 'split' (composite), or 'both'
        self.camera_display_mode = 'split'  # 'window', 'split', or 'both'
        
        # Simulation timing
        self.simulation_speed = SIMULATION_SPEED  # Speed multiplier (0.1-2.0)
        self.initial_fps = GUI_FPS  # Store original FPS for reference
        
        # Kick camera into a reasonable initial pose
        self._update_camera()
        self._lidar_indicator = {"handles": [], "update_every": 1, "last_update": 0}

    # ---------------- scene build helpers ----------------

    def _valid(self, x, y, min_sep=0.8):
        """Placement rejection test: avoids heavy overlap and respects the center clearing."""
        if self.clearing and math.hypot(x, y) < self.clearing_radius:
            return False
        for ox, oy, r in self.placed:
            if dist((x, y), (ox, oy)) < (r + min_sep):
                return False
        return True

    def _setup_ground(self):
        """Create the ground plane and apply a texture if available."""
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        try:
            tex = p.loadTexture("grass.png", physicsClientId=self.client)  # pybullet_data contains grass.png
            p.changeVisualShape(plane_id, -1, textureUniqueId=tex, physicsClientId=self.client)
        except Exception:
            # Texture may not exist in all setups; keep a reasonable fallback.
            p.changeVisualShape(plane_id, -1, rgbaColor=[0.35, 0.45, 0.25, 1.0], physicsClientId=self.client)

    def _place_obstacles(self):
        """Spawn a small set of simple static obstacles (pillars, short walls, crates)."""
        for i in range(self.n_obstacles):
            for _ in range(40):
                x, y = rand_in_disk(self.radius * 0.93)
                if not self._valid(x, y, min_sep=1.8):
                    continue

                typ = i % 3
                if typ == 0:
                    # Tall pillar
                    height = random.uniform(1.6, 3.8)
                    radius = random.uniform(0.22, 0.48)
                    color = random.choice(
                        [[0.6, 0.6, 0.65, 1], [0.5, 0.5, 0.55, 1], [0.7, 0.7, 0.72, 1]]
                    )
                    coll = p.createCollisionShape(
                        p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.client
                    )
                    vis = p.createVisualShape(
                        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=self.client
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=coll,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, height / 2.0],
                        physicsClientId=self.client,
                    )
                    self.placed.append((x, y, max(radius, 0.6)))

                elif typ == 1:
                    # Short wall segment
                    length = random.uniform(1.0, 3.0)
                    thickness = random.uniform(0.12, 0.30)
                    height = random.uniform(0.6, 1.6)
                    yaw = random.uniform(0, math.pi)
                    color = random.choice([[0.65, 0.25, 0.2, 1], [0.55, 0.27, 0.22, 1]])
                    coll = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[length / 2, thickness / 2, height / 2],
                        physicsClientId=self.client,
                    )
                    vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[length / 2, thickness / 2, height / 2],
                        rgbaColor=color,
                        physicsClientId=self.client,
                    )
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=coll,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, height / 2.0],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )
                    self.placed.append((x, y, max(length / 2, 0.8)))

                else:
                    # Box/crate
                    side = random.uniform(0.4, 1.2)
                    height = random.uniform(0.4, 1.0)
                    color = random.choice([[0.3, 0.3, 0.35, 1], [0.45, 0.35, 0.25, 1]])
                    coll = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[side / 2, side / 2, height / 2],
                        physicsClientId=self.client,
                    )
                    vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[side / 2, side / 2, height / 2],
                        rgbaColor=color,
                        physicsClientId=self.client,
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=coll,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, height / 2.0],
                        physicsClientId=self.client,
                    )
                    self.placed.append((x, y, side / 1.8))

                break

    def _place_buildings(self):
        """Spawn a few building-sized boxes and optionally add a simple roof."""
        for _ in range(self.n_buildings):
            for _ in range(40):
                x, y = rand_in_disk(self.radius * 0.85)  # keep away from the boundary
                if not self._valid(x, y, min_sep=3.0):  # buildings need more breathing room
                    continue

                width = random.uniform(3.0, 6.0)
                length = random.uniform(3.0, 6.0)
                height = random.uniform(2.5, 4.5)

                color = random.choice(
                    [
                        [0.9, 0.85, 0.7, 1],  # Beige
                        [0.95, 0.95, 0.95, 1],  # White
                        [0.7, 0.7, 0.75, 1],  # Grey
                    ]
                )

                coll = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[width / 2, length / 2, height / 2],
                    physicsClientId=self.client,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[width / 2, length / 2, height / 2],
                    rgbaColor=color,
                    physicsClientId=self.client,
                )

                yaw = random.uniform(0, math.pi)
                orn = p.getQuaternionFromEuler([0, 0, yaw])
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=coll,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, height / 2.0],
                    baseOrientation=orn,
                    physicsClientId=self.client,
                )

                # Roof is purely visual geometry (still static)
                if random.random() < 0.7:
                    roof_height = random.uniform(0.5, 1.0)
                    roof_color = [0.6, 0.3, 0.2, 1]
                    roof_coll1 = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2 + 0.2, length / 4, roof_height / 2],
                        physicsClientId=self.client,
                    )
                    roof_vis1 = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2 + 0.2, length / 4, roof_height / 2],
                        rgbaColor=roof_color,
                        physicsClientId=self.client,
                    )
                    roof_coll2 = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2 + 0.2, length / 4, roof_height / 2],
                        physicsClientId=self.client,
                    )
                    roof_vis2 = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2 + 0.2, length / 4, roof_height / 2],
                        rgbaColor=roof_color,
                        physicsClientId=self.client,
                    )
                    roof_y1 = y + length / 4
                    roof_y2 = y - length / 4
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=roof_coll1,
                        baseVisualShapeIndex=roof_vis1,
                        basePosition=[x, roof_y1, height + roof_height / 2],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=roof_coll2,
                        baseVisualShapeIndex=roof_vis2,
                        basePosition=[x, roof_y2, height + roof_height / 2],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )
                else:
                    roof_thickness = 0.2
                    roof_color = [0.4, 0.4, 0.4, 1]
                    roof_coll = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2, length / 2, roof_thickness / 2],
                        physicsClientId=self.client,
                    )
                    roof_vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[width / 2, length / 2, roof_thickness / 2],
                        rgbaColor=roof_color,
                        physicsClientId=self.client,
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=roof_coll,
                        baseVisualShapeIndex=roof_vis,
                        basePosition=[x, y, height + roof_thickness / 2],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )

                self.placed.append((x, y, max(width, length) * 0.7))
                break

    def _place_trees(self):
        """Scatter trees made of a trunk cylinder plus a small crown of spheres."""
        attempts = 0
        placed_count = 0
        max_attempts = self.n_trees * 50
        while placed_count < self.n_trees and attempts < max_attempts:
            attempts += 1
            x, y = rand_in_disk(self.radius * 0.9)
            if not self._valid(x, y, min_sep=0.9):
                continue
            trunk_h = random.uniform(1.6, 3.2)
            trunk_r = random.uniform(0.12, 0.26)
            crown = random.uniform(0.7, 1.6)

            trunk_col = [
                0.4 + random.uniform(-0.05, 0.05),
                0.3 + random.uniform(-0.04, 0.04),
                0.2 + random.uniform(-0.02, 0.02),
                1.0,
            ]
            crown_col = random.choice(
                [
                    [0.85, 0.3, 0.1, 1.0],
                    [0.9, 0.5, 0.1, 1.0],
                    [0.8, 0.7, 0.1, 1.0],
                ]
            )

            self._add_tree(x, y, trunk_h, trunk_r, crown, trunk_col, crown_col)
            body_radius = crown * 1.05
            self.placed.append((x, y, body_radius))
            placed_count += 1

    def _add_tree(self, x, y, trunk_h, trunk_r, crown_size, trunk_col, crown_col):
        """Create one tree (trunk + crown) at the given position."""
        trunk_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h, physicsClientId=self.client
        )
        trunk_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h, rgbaColor=trunk_col, physicsClientId=self.client
        )
        trunk_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_collision,
            baseVisualShapeIndex=trunk_visual,
            basePosition=[x, y, trunk_h / 2.0],
            physicsClientId=self.client,
        )
        try:
            p.changeVisualShape(trunk_body, -1, specularColor=[0.06, 0.06, 0.06], physicsClientId=self.client)
        except Exception:
            pass

        n_spheres = random.randint(2, 4)
        for i in range(n_spheres):
            offset_x = np.random.normal(scale=0.12)
            offset_y = np.random.normal(scale=0.12)
            sphere_radius = crown_size * random.uniform(0.65, 1.05) * (1.0 - i * 0.07)
            z = trunk_h + i * (sphere_radius * 0.55) + 0.02 + random.uniform(-0.02, 0.02)
            coll = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius, physicsClientId=self.client)
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=sphere_radius,
                rgbaColor=[
                    crown_col[0] * (0.9 + random.random() * 0.15),
                    min(1.0, crown_col[1] * (0.9 + random.random() * 0.15)),
                    crown_col[2] * (0.9 + random.random() * 0.15),
                    1.0,
                ],
                physicsClientId=self.client,
            )
            crown_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=coll,
                baseVisualShapeIndex=vis,
                basePosition=[x + offset_x, y + offset_y, z],
                physicsClientId=self.client,
            )
            try:
                p.changeVisualShape(crown_body, -1, specularColor=[0.06, 0.06, 0.06], physicsClientId=self.client)
            except Exception:
                pass

    def _place_bushes_and_grass(self):
        """Add low-profile foliage: bushes plus a small amount of grass."""
        placed = 0
        attempts = 0
        while placed < self.n_bushes and attempts < self.n_bushes * 20:
            attempts += 1
            x, y = rand_in_disk(self.radius * 0.95)
            if not self._valid(x, y, min_sep=0.35):
                continue
            size = random.uniform(0.18, 0.46)
            color = [
                0.7 + random.random() * 0.2,
                0.4 + random.random() * 0.3,
                0.1 + random.random() * 0.1,
                1.0,
            ]
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color, physicsClientId=self.client)
            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, size * 0.45],
                physicsClientId=self.client,
            )
            try:
                p.changeVisualShape(body, -1, specularColor=[0.05, 0.05, 0.05], physicsClientId=self.client)
            except Exception:
                pass
            self.placed.append((x, y, size))
            placed += 1

        for _ in range(self.n_grass):
            x, y = rand_in_disk(self.radius * 0.98)
            if not self._valid(x, y, min_sep=0.03):
                continue
            h = random.uniform(0.05, 0.18)
            w = random.uniform(0.02, 0.04)
            color = [
                0.6 + random.random() * 0.2,
                0.5 + random.random() * 0.2,
                0.1 + random.random() * 0.1,
                1.0,
            ]
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[w, w * 1.2, h],
                rgbaColor=color,
                physicsClientId=self.client,
            )
            yaw = random.random() * 2 * math.pi
            orn = p.getQuaternionFromEuler([random.uniform(-0.12, 0.12), 0.0, yaw])
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, h],
                baseOrientation=orn,
                physicsClientId=self.client,
            )

    def _place_rocks(self):
        """Scatter rocks/boulders with a bit of size/shape variation."""
        for _ in range(self.n_rocks):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.9)
                if not self._valid(x, y, min_sep=0.15):
                    continue

                if random.random() < 0.6:
                    r = random.uniform(0.12, 0.45)
                    sx = random.uniform(0.7, 1.3)
                    sy = random.uniform(0.7, 1.3)
                    sz = random.uniform(0.6, 1.0)
                    coll = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[r * sx, r * sy, r * sz], physicsClientId=self.client
                    )
                    vis = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=r,
                        rgbaColor=[
                            0.45 + random.random() * 0.18,
                            0.45 + random.random() * 0.18,
                            0.42 + random.random() * 0.12,
                            1,
                        ],
                        physicsClientId=self.client,
                    )
                    yaw = random.random() * 2 * math.pi
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    body = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=coll,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, r * 0.8],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )
                    try:
                        p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                    except Exception:
                        pass
                    self.placed.append((x, y, r * max(sx, sy) * 1.05))
                    break

                hx = random.uniform(0.08, 0.6)
                hy = random.uniform(0.08, 0.6)
                hz = random.uniform(0.05, 0.32)
                coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client)
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[hx, hy, hz],
                    rgbaColor=[
                        0.4 + random.random() * 0.24,
                        0.4 + random.random() * 0.18,
                        0.38 + random.random() * 0.16,
                        1,
                    ],
                    physicsClientId=self.client,
                )
                yaw = random.random() * 2 * math.pi
                orn = p.getQuaternionFromEuler([random.uniform(-0.18, 0.18), random.uniform(-0.12, 0.12), yaw])
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=coll,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, hz],
                    baseOrientation=orn,
                    physicsClientId=self.client,
                )
                try:
                    p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                except Exception:
                    pass
                self.placed.append((x, y, max(hx, hy) * 1.05))
                break

    def _place_logs(self):
        """Place a few fallen logs (cylinders rotated to look like they fell over)."""
        for _ in range(self.n_logs):
            attempts = 0
            while attempts < 8:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.9)
                if not self._valid(x, y, min_sep=0.9):
                    continue
                length = random.uniform(0.8, 2.0)
                radius = random.uniform(0.075, 0.17)
                coll = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=self.client
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=length,
                    rgbaColor=[
                        0.33 + random.random() * 0.06,
                        0.20 + random.random() * 0.04,
                        0.10 + random.random() * 0.03,
                        1,
                    ],
                    physicsClientId=self.client,
                )
                yaw = random.random() * 2 * math.pi
                pitch = random.uniform(-0.35, 0.35)
                orn = p.getQuaternionFromEuler([pitch, 0, yaw])
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=coll,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, radius * 0.45],
                    baseOrientation=orn,
                    physicsClientId=self.client,
                )
                try:
                    p.changeVisualShape(body, -1, specularColor=[0.04, 0.04, 0.04], physicsClientId=self.client)
                except Exception:
                    pass
                self.placed.append((x, y, length * 0.6))
                break

    def _place_mushrooms(self):
        """Add small mushroom clusters near the ground."""
        for _ in range(self.n_mushrooms):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.95)
                if not self._valid(x, y, min_sep=0.15):
                    continue
                cluster_size = random.randint(2, 8)
                for _ in range(cluster_size):
                    dx = np.random.normal(scale=0.10)
                    dy = np.random.normal(scale=0.10)
                    stem_h = random.uniform(0.05, 0.12)
                    stem_r = random.uniform(0.01, 0.03)
                    cap_r = random.uniform(0.03, 0.07)
                    stem_vis = p.createVisualShape(
                        p.GEOM_CYLINDER,
                        radius=stem_r,
                        length=stem_h,
                        rgbaColor=[0.95, 0.9, 0.82, 1],
                        physicsClientId=self.client,
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=stem_vis,
                        basePosition=[x + dx, y + dy, stem_h / 2.0],
                        physicsClientId=self.client,
                    )
                    cap_vis = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=cap_r,
                        rgbaColor=[
                            0.9,
                            0.2 + random.random() * 0.3,
                            0.18 + random.random() * 0.18,
                            1,
                        ],
                        physicsClientId=self.client,
                    )
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=cap_vis,
                        basePosition=[x + dx, y + dy, stem_h + cap_r * 0.28],
                        physicsClientId=self.client,
                    )
                self.placed.append((x, y, 0.2))
                break

    def _place_leaf_clumps(self):
        """Scatter tiny low-height clutter to add detail on the ground."""
        for _ in range(self.leaf_clumps):
            attempts = 0
            while attempts < 6:
                attempts += 1
                x, y = rand_in_disk(self.radius * 0.95)
                if not self._valid(x, y, min_sep=0.07):
                    continue
                count = random.randint(6, 20)
                for _ in range(count):
                    dx = np.random.normal(scale=0.18)
                    dy = np.random.normal(scale=0.18)
                    w = random.uniform(0.02, 0.06)
                    h = 0.003 + random.uniform(0.0, 0.01)
                    color = [
                        0.25 + random.random() * 0.35,
                        0.35 + random.random() * 0.35,
                        0.06 + random.random() * 0.2,
                        1,
                    ]
                    vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[w, w * 1.2, h],
                        rgbaColor=color,
                        physicsClientId=self.client,
                    )
                    yaw = random.random() * 2 * math.pi
                    orn = p.getQuaternionFromEuler([0, 0, yaw])
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=vis,
                        basePosition=[x + dx, y + dy, h],
                        baseOrientation=orn,
                        physicsClientId=self.client,
                    )
                self.placed.append((x, y, 0.25))
                break

    def _place_center_marker(self):
        """Draw a small 'START' marker near the origin so orientation is obvious."""
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.08, 0.7, 0.08, 0.95], physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.08], physicsClientId=self.client
        )
        try:
            p.addUserDebugText(
                "START",
                [0.0, -0.8, 0.02],
                textColorRGB=[0.02, 0.6, 0.02],
                textSize=1.2,
                physicsClientId=self.client,
            )
        except Exception:
            pass

    def _add_boundary_direction_labels(self):
        """Draw compass labels near the arena edge for quick orientation."""
        label_height = 2.0
        label_distance = self.radius - 2.0

        # North (+X)
        p.addUserDebugText(
            "NORTH",
            [label_distance, 0, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.client,
        )

        # South (-X)
        p.addUserDebugText(
            "SOUTH",
            [-label_distance, 0, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.client,
        )

        # East (+Y)
        p.addUserDebugText(
            "EAST",
            [0, label_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.client,
        )

        # West (-Y)
        p.addUserDebugText(
            "WEST",
            [0, -label_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.client,
        )

        diag_distance = label_distance * 0.7

        p.addUserDebugText(
            "NORTH-EAST",
            [diag_distance, diag_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0,
            physicsClientId=self.client,
        )
        p.addUserDebugText(
            "NORTH-WEST",
            [diag_distance, -diag_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0,
            physicsClientId=self.client,
        )
        p.addUserDebugText(
            "SOUTH-EAST",
            [-diag_distance, diag_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0,
            physicsClientId=self.client,
        )
        p.addUserDebugText(
            "SOUTH-WEST",
            [-diag_distance, -diag_distance, label_height],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0,
            physicsClientId=self.client,
        )

    # ---------------- drone + camera helpers ----------------

    def _load_drone(self):
        """Spawn a drone at the origin (URDF if found, otherwise a placeholder)."""
        # Candidate URDF locations (some are optional depending on your setup)
        drone_urdf_paths = []
        
        # Try gym_pybullet_drones (if installed)
        try:
            gym_drones_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/cf2x.urdf')
            drone_urdf_paths.append(gym_drones_path)
        except (ImportError, ModuleNotFoundError, Exception):
            # Not installed / not available: fall back to other paths.
            pass
        
        # Common relative paths used in this workspace
        drone_urdf_paths.extend([
            '../gym-pybullet-drones 2/gym_pybullet_drones/assets/cf2x.urdf',
            '../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf',
            '../../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf',
            # Absolute-ish via this file location
            os.path.join(os.path.dirname(__file__), '../testing-Agents/gym_pybullet_drones/assets/cf2x.urdf'),
        ])
        
        drone_urdf = None
        for path in drone_urdf_paths:
            if path and os.path.exists(path):
                drone_urdf = path
                break
        
        if drone_urdf is None:
            print("Warning: Could not find cf2x.urdf. Creating a simple drone placeholder.")
            # Placeholder: a small body plus four propeller spheres
            self.drone_id = self._create_simple_drone([0, 0, 1.0])
        else:
            print(f"Loading drone from: {drone_urdf}")
            init_position = [0, 0, 1.0]
            init_orientation = p.getQuaternionFromEuler([0, 0, 0])

            self.drone_id = p.loadURDF(
                drone_urdf,
                init_position,
                init_orientation,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.client,
            )
        
        # Track drone state (used by camera and control)
        self.drone_position = np.array([0, 0, 1.0])
        self.drone_velocity = np.array([0, 0, 0])
        
        # Simple movement targets (not a real drone controller)
        self.hover_height = 1.5
        self.target_position = np.array([0, 0, self.hover_height])
        self._start_time = None  # Initialize start time for movement pattern

    def _create_simple_drone(self, position):
        """Build a tiny placeholder drone model when no URDF is available."""
        scale = getattr(self, 'drone_scale', 2.0)  # Use scale factor if available
        
        # Body
        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05 * scale, 0.05 * scale, 0.02 * scale],
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=self.client
        )
        body_coll = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05 * scale, 0.05 * scale, 0.02 * scale],
            physicsClientId=self.client
        )
        
        body_id = p.createMultiBody(
            baseMass=0.027 * (scale ** 3),  # Mass scales with volume
            baseCollisionShapeIndex=body_coll,
            baseVisualShapeIndex=body_vis,
            basePosition=position,
            physicsClientId=self.client
        )
        
        # Propellers (visual only)
        prop_positions = [
            [0.04 * scale, 0.04 * scale, 0.03 * scale],
            [-0.04 * scale, 0.04 * scale, 0.03 * scale],
            [0.04 * scale, -0.04 * scale, 0.03 * scale],
            [-0.04 * scale, -0.04 * scale, 0.03 * scale]
        ]
        prop_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02 * scale,
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
            physicsClientId=self.client
        )
        
        for prop_pos in prop_positions:
            prop_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=prop_vis,
                basePosition=[position[0] + prop_pos[0], position[1] + prop_pos[1], position[2] + prop_pos[2]],
                physicsClientId=self.client
            )
        
        return body_id

    def _handle_camera_input(self):
        """Handle camera hotkeys: arrows = orbit, +/- = zoom, 'l' = lidar overlay."""
        try:
            keys = p.getKeyboardEvents(physicsClientId=self.client)
        except Exception:
            return

        delta_yaw = 3.0
        delta_pitch = 2.0
        delta_zoom = 0.3

        if keys.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN:
            self._camera_yaw -= delta_yaw
        if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN:
            self._camera_yaw += delta_yaw
        if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN:
            self._camera_pitch = max(-89.0, self._camera_pitch - delta_pitch)
        if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN:
            self._camera_pitch = min(-5.0, self._camera_pitch + delta_pitch)

        zoom_in_codes = [ord('+'), ord('='), getattr(p, "B3G_NUMPAD_PLUS", None)]
        zoom_out_codes = [ord('-'), ord('_'), getattr(p, "B3G_NUMPAD_MINUS", None)]

        if any(code is not None and keys.get(code, 0) & p.KEY_IS_DOWN for code in zoom_in_codes):
            self.camera_distance = max(0.8, self.camera_distance - delta_zoom)
        if any(code is not None and keys.get(code, 0) & p.KEY_IS_DOWN for code in zoom_out_codes):
            self.camera_distance = min(25.0, self.camera_distance + delta_zoom)

        if keys.get(ord('l'), 0) & p.KEY_WAS_TRIGGERED:
            self.show_lidar_overlay = not self.show_lidar_overlay
            state = "ON" if self.show_lidar_overlay else "OFF"
            print(f"Lidar overlay: {state}")
            if not self.show_lidar_overlay:
                indicator = getattr(self, "_lidar_indicator", None)
                if indicator and indicator.get("handles"):
                    for handle in indicator["handles"]:
                        try:
                            p.removeUserDebugItem(handle, physicsClientId=self.client)
                        except Exception:
                            pass
                    indicator["handles"] = []

    def _update_camera(self):
        """Update the debug visualizer camera to follow the drone."""
        if not hasattr(self, 'drone_id') or self.client is None:
            return

        try:
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)

            self.drone_position = np.array(pos)
            self.drone_velocity = np.array(vel)

            self._handle_camera_input()

            camera_target = pos
            velocity_magnitude = np.linalg.norm(self.drone_velocity)
            if velocity_magnitude > 0.1:
                movement_direction = math.degrees(math.atan2(self.drone_velocity[1], self.drone_velocity[0]))
                auto_yaw = movement_direction + 180.0
                blend = 0.9
                self._camera_yaw = blend * self._camera_yaw + (1 - blend) * auto_yaw

            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
                cameraTargetPosition=camera_target,
                physicsClientId=self.client,
            )
        except Exception:
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=12,
                    cameraYaw=45,
                    cameraPitch=-35,
                    cameraTargetPosition=[0, 0, 0],
                    physicsClientId=self.client,
                )
            except Exception:
                pass

    def _render_lidar_overlay(self, drone_pos, drone_yaw, lidar_hits):
        """Draw lidar rays around the drone using debug lines."""
        if not getattr(self, "show_lidar_overlay", False):
            return
        indicator = getattr(self, "_lidar_indicator", None)
        if indicator is None or self.client is None:
            return

        indicator["last_update"] += 1
        if indicator["last_update"] < indicator["update_every"]:
            return
        indicator["last_update"] = 0

        try:
            if indicator["handles"]:
                for handle in indicator["handles"]:
                    p.removeUserDebugItem(handle, physicsClientId=self.client)
                indicator["handles"] = []
        except Exception:
            pass

        safe_color = [0.1, 0.8, 0.1]
        warn_color = [0.9, 0.6, 0.1]
        danger_color = [0.9, 0.1, 0.1]

        lidar_range = getattr(self, "lidar_range", 2.0)
        angle = drone_yaw
        for frac in lidar_hits:
            dist = frac * lidar_range
            endpoint = [
                drone_pos[0] + dist * math.cos(angle),
                drone_pos[1] + dist * math.sin(angle),
                drone_pos[2],
            ]

            color = safe_color
            if dist < 1.0:
                color = danger_color
            elif dist < 2.0:
                color = warn_color

            try:
                handle = p.addUserDebugLine(
                    lineFromXYZ=drone_pos,
                    lineToXYZ=endpoint,
                    lineColorRGB=color,
                    lineWidth=2.0,
                    lifeTime=0.2,
                    physicsClientId=self.client,
                )
                indicator["handles"].append(handle)
            except Exception:
                break

            angle += 2 * math.pi / len(lidar_hits)

    def _control_drone(self):
        """Very simple motion: hover + a slow circular trajectory."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
            
            self.drone_position = np.array(pos)
            self.drone_velocity = np.array(vel)
            
            # Basic P-D style height stabilization
            height_error = self.hover_height - pos[2]
            desired_force_z = height_error * 5.0 - vel[2] * 2.0  # P-D control
            
            # Add gentle horizontal motion (circle)
            time_step = p.getPhysicsEngineParameters(physicsClientId=self.client)['fixedTimeStep']
            # Initialize start time once
            if self._start_time is None:
                self._start_time = time.time()
                
                t = 0.0
            else:
                t = time.time() - self._start_time
            
            # Circle parameters
            radius = 3.0
            angular_vel = 0.3
            target_x = radius * math.cos(angular_vel * t)
            target_y = radius * math.sin(angular_vel * t)
            target_z = self.hover_height
            
            # Drive towards the target point
            pos_error = np.array([target_x, target_y, target_z]) - self.drone_position
            desired_force = pos_error * 2.0 - self.drone_velocity * 1.0
            
            # Apply forces directly (not rotor physics)
            max_force = 10.0
            force = np.clip(desired_force, -max_force, max_force)
            
            # Main force for translation
            p.applyExternalForce(
                self.drone_id,
                -1,  # -1 means apply to base
                force.tolist(),
                [0, 0, 0],  # position (relative to COM, 0 means at COM)
                p.WORLD_FRAME,
                physicsClientId=self.client
            )
            
            # Upward force to counter gravity (estimate mass if needed)
            drone_mass = p.getDynamicsInfo(self.drone_id, -1, physicsClientId=self.client)[0]
            if drone_mass > 0:
                hover_force = 9.81 * drone_mass
            else:
                # Placeholder mass estimate
                scale = getattr(self, 'drone_scale', 2.0)
                hover_force = 9.81 * 0.027 * (scale ** 3)
            
            p.applyExternalForce(
                self.drone_id,
                -1,
                [0, 0, hover_force],
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=self.client
            )
            
        except Exception as e:
            print(f"Error controlling drone: {e}")

    def _get_drone_camera_image(self):
        """Capture a simple first-person RGB image from the drone pose."""
        if not hasattr(self, 'drone_id') or not self.show_drone_camera:
            return None
        
        try:
            # Pose
            pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
            
            # Quaternion -> rotation matrix
            quat = orn
            rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            
            # Camera pose relative to drone
            camera_offset = np.array([0.1, 0, 0.05])  # Forward (x), right (y), up (z)
            camera_pos = pos + np.dot(rot_matrix, camera_offset)
            
            # Look forward
            forward_vec = np.dot(rot_matrix, np.array([1, 0, 0]))  # Forward direction
            target_pos = camera_pos + forward_vec * 5.0  # Look 5m ahead
            
            # Up vector
            up_vec = np.dot(rot_matrix, np.array([0, 0, 1]))  # Up direction
            
            # View/projection
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=up_vec.tolist(),
                physicsClientId=self.client
            )
            
            # Compute projection matrix
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.drone_camera_fov,
                aspect=self.drone_camera_width / self.drone_camera_height,
                nearVal=0.01,
                farVal=100.0,
                physicsClientId=self.client
            )
            
            # Render
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=self.drone_camera_width,
                height=self.drone_camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client
            )
            
            # PyBullet returns RGBA; reshape and drop alpha.
            rgb_array = np.array(rgb_img, dtype=np.uint8)
            
            # Handle possible return formats
            if rgb_array.ndim == 1:
                # Flat array -> (H, W, 4)
                rgb_array = rgb_array.reshape((self.drone_camera_height, self.drone_camera_width, 4))
            elif rgb_array.ndim != 3:
                print(f"Warning: Unexpected camera image shape: {rgb_array.shape}, skipping frame")
                return None
                
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel (keep only RGB)
            rgb_array = np.flipud(rgb_array)  # PyBullet image is upside-down
            
            return rgb_array
            
        except Exception as e:
            print(f"Error capturing drone camera: {e}")
            return None

    def _display_drone_camera_opencv(self, img):
        """Show the drone camera in a dedicated OpenCV window."""
        if img is None or not CV2_AVAILABLE:
            return
        
        try:
            # Upscale a bit for readability
            display_img = cv2.resize(img, (640, 480))
            
            # Lightweight overlay text
            cv2.putText(display_img, "DRONE CAMERA VIEW", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, "Press 'c' to toggle camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Present
            cv2.imshow("Drone Camera", display_img)
            cv2.waitKey(1)  # Required for OpenCV to update the window
            
        except Exception as e:
            print(f"Error displaying drone camera in OpenCV: {e}")

    def _display_drone_camera(self, img):
        """Display the drone camera based on the selected display mode."""
        if img is None:
            return
        
        # Window mode
        if self.camera_display_mode in ['window', 'both'] and CV2_AVAILABLE:
            self._display_drone_camera_opencv(img)
        
        # Split mode: OpenCV composite window (PyBullet doesn't provide split overlays)
        if self.camera_display_mode in ['split', 'both'] and CV2_AVAILABLE:
            try:
                # Left: placeholder for main PyBullet view, Right: drone camera
                
                # Resize for the split layout
                drone_img_resized = cv2.resize(img, (640, 480))
                
                # Border + label
                cv2.rectangle(drone_img_resized, (0, 0), (drone_img_resized.shape[1]-1, drone_img_resized.shape[0]-1), (0, 255, 0), 3)
                cv2.putText(drone_img_resized, "DRONE CAMERA VIEW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(drone_img_resized, "Press 'c' to toggle", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Placeholder image for the main PyBullet view
                main_view = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(main_view, "PYBULLET MAIN VIEW", (150, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(main_view, "(Check PyBullet window)", (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Combine side-by-side
                split_view = np.hstack([main_view, drone_img_resized])
                
                # Present composite
                cv2.imshow("Split View - Main | Drone Camera", split_view)
                cv2.waitKey(1)
                
            except Exception as e:
                print(f"Error displaying split-screen view: {e}")

    def relocate_humans(self, target_direction=None):
        """Relocate humans (optionally biased along a direction vector)."""
        if not self.human_ids:
            self._place_humans()
            return
        if target_direction:
            self.human_target_direction = target_direction
        for idx, human_id in enumerate(self.human_ids):
            pos = self._sample_human_spot()
            self.human_positions[idx] = pos
            p.resetBasePositionAndOrientation(human_id, pos, [0, 0, 0, 1], physicsClientId=self.client)
            self._update_human_label(idx, pos)

    def reposition_humans(self, target_direction=None):
        """Backward-compatible name for relocate_humans()."""
        return self.relocate_humans(target_direction=target_direction)

    def _place_humans(self):
        """Spawn humans and attach simple debug labels."""
        self._clear_human_labels()
        self.human_ids = []
        self.human_positions = []
        self.human_label_ids = []
        for i in range(self.num_humans):
            pos = self._sample_human_spot()
            if pos is None:
                continue
            human_id = self._create_human_body(pos, HUMAN_COLOR)
            if human_id is not None:
                self.human_ids.append(human_id)
                self.human_positions.append(pos)
                self.human_label_ids.append(self._create_human_label(pos))

    def _sample_human_spot(self):
        """Choose a human position (optionally along the configured target direction)."""
        for _ in range(60):
            if self.human_target_direction:
                # Sample along the direction vector
                angle = math.atan2(self.human_target_direction[1], self.human_target_direction[0])
                # Vary distance but keep heading
                r = self.radius * random.uniform(0.4, 0.9)  # Further out towards boundary
                x = r * math.cos(angle)
                y = r * math.sin(angle)
            else:
                x, y = rand_in_disk(self.radius * 0.85)
            if self._valid(x, y, min_sep=0.8):
                return [x, y, HUMAN_HEIGHT / 2.0]
        return None

    def _create_human_body(self, position, color):
        """Create a minimal human proxy (a cylinder)."""
        coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=HUMAN_RADIUS, height=HUMAN_HEIGHT, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=HUMAN_RADIUS, length=HUMAN_HEIGHT, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                  basePosition=position, physicsClientId=self.client)

    def _create_human_label(self, position):
        """Attach a single-character debug label above the human."""
        return p.addUserDebugText("H", [position[0], position[1], position[2] + HUMAN_HEIGHT/2 + 0.3],
                                  textColorRGB=[1.0, 1.0, 0.2], textSize=1.4, lifeTime=0, physicsClientId=self.client)

    def _update_human_label(self, idx, position):
        """Move the debug label to match a human's new pose."""
        if idx < len(self.human_label_ids):
            # Remove existing label, then re-create at new position
            try:
                p.removeUserDebugItem(self.human_label_ids[idx], physicsClientId=self.client)
            except:
                pass
            self.human_label_ids[idx] = self._create_human_label(position)

    def _clear_human_labels(self):
        """Remove all human debug labels from the scene."""
        for label_id in self.human_label_ids:
            try:
                p.removeUserDebugItem(label_id, physicsClientId=self.client)
            except:
                pass
        self.human_label_ids = []

    def run(self, fps=GUI_FPS):
        """Run the simulation loop until quit is requested."""
        base_dt = 1.0 / fps  # base timestep
        dt = base_dt / self.simulation_speed  # speed-adjusted timestep
        
        print("Scene running. Press ESC or 'q' in the PyBullet window or close the window to exit.")
        print("Drone is flying with a following camera view.")
        print(f"Simulation speed: {self.simulation_speed}x ({'SLOW' if self.simulation_speed < 0.5 else 'NORMAL' if self.simulation_speed < 1.5 else 'FAST'})")
        print("Controls:")
        print("  Press 'c' to toggle drone camera")

    def start(self, fps=GUI_FPS):
        """Preferred entrypoint for the interactive scene loop."""
        return self.run(fps=fps)
        print("  Press 'm' to change display mode")
        print("  Press '+' or '=' to speed up simulation")
        print("  Press '-' or '_' to slow down simulation")
        print("  Press '0' to reset to normal speed")
        
        if CV2_AVAILABLE and self.show_drone_camera:
            print("Drone camera view enabled. Press 'c' in PyBullet window to toggle.")
            print(f"Display mode: {self.camera_display_mode}")
            if self.camera_display_mode == 'split':
                print("Split-screen view: Check OpenCV window for side-by-side display")
            elif self.camera_display_mode == 'window':
                print("Separate window: Check 'Drone Camera' OpenCV window")
            elif self.camera_display_mode == 'both':
                print("Both modes: Check both OpenCV windows")
        elif not CV2_AVAILABLE:
            print("Warning: OpenCV not available. Install with: pip install opencv-python")
        try:
            while True:
                if not p.isConnected(physicsClientId=self.client):
                    break
                keys = p.getKeyboardEvents()

                # Quit keys
                if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                    print("ESC pressed. Exiting.")
                    break
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    print("'q' pressed. Exiting.")
                    break
                
                # Toggle camera overlay
                if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                    self.show_drone_camera = not self.show_drone_camera
                    if self.show_drone_camera:
                        print("Drone camera view: ENABLED")
                        print(f"  Display mode: {self.camera_display_mode}")
                    else:
                        print("Drone camera view: DISABLED")
                        if CV2_AVAILABLE:
                            try:
                                cv2.destroyWindow("Drone Camera")
                                cv2.destroyWindow("Split View - Main | Drone Camera")
                            except:
                                pass
                
                # Cycle camera display mode
                if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED and CV2_AVAILABLE:
                    modes = ['split', 'window', 'both']
                    current_idx = modes.index(self.camera_display_mode) if self.camera_display_mode in modes else 0
                    next_idx = (current_idx + 1) % len(modes)
                    self.camera_display_mode = modes[next_idx]
                    print(f"Display mode changed to: {self.camera_display_mode}")
                
                # Speed control
                if ord('+') in keys and keys[ord('+')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = min(2.0, self.simulation_speed + 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('=') in keys and keys[ord('=')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = min(2.0, self.simulation_speed + 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('_') in keys and keys[ord('_')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed: {self.simulation_speed:.1f}x")
                if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
                    self.simulation_speed = 1.0
                    dt = base_dt / self.simulation_speed
                    print(f"Simulation speed reset to: {self.simulation_speed:.1f}x (normal)")

                # Update drone motion
                if hasattr(self, 'drone_id'):
                    self._control_drone()
                
                # Follow camera
                self._update_camera()
                
                # Drone camera rendering (rate-limited)
                if self.show_drone_camera and CV2_AVAILABLE:
                    self.drone_camera_frame_count += 1
                    if self.drone_camera_frame_count >= self.drone_camera_update_freq:
                        drone_img = self._get_drone_camera_image()
                        if drone_img is not None:
                            self._display_drone_camera(drone_img)
                        self.drone_camera_frame_count = 0

                # Step simulation and sleep
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(dt)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            # Cleanup
            if CV2_AVAILABLE:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception:
                pass
            print("Disconnected PyBullet.")


if __name__ == "__main__":
    scene = EnhancedForestWithObstacles(
        gui=KEEP_GUI,
        radius=ARENA_RADIUS,
        n_trees=NUM_TREES,
        n_bushes=NUM_BUSHES,
        n_grass=NUM_GRASS,
        n_rocks=NUM_ROCKS,
        n_logs=NUM_LOGS,
        n_mushrooms=NUM_MUSHROOM_CLUSTERS,
        leaf_clumps=LEAF_CLUMPS,
        n_obstacles=NUM_OBSTACLES,
        n_buildings=NUM_BUILDINGS,
        clearing=CLEARING,
        clearing_radius=CLEARING_RADIUS
    )
    scene.start(fps=GUI_FPS)
