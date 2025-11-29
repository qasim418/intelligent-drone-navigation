import pybullet as p
import pybullet_data
import time

# ==============================================================================
# EMERGENCY MARKER FUNCTIONS
# ==============================================================================
def add_human_marker(position, text="SURVIVOR"):
    """Add a red human figure at target position"""
    # Red sphere for head
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
    body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, 
                                basePosition=position)
    # Floating label
    p.addUserDebugText(text, [position[0], position[1], position[2] + 1.5], 
                       textColorRGB=[1, 0, 0], textSize=1.2)
    return body_id

def add_fire_marker(position, text="FIRE"):
    """Add an orange fire marker"""
    vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.0, 
                                 rgbaColor=[1, 0.3, 0, 0.9])
    fire_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, 
                                basePosition=[position[0], position[1], position[2] + 0.5])
    p.addUserDebugText(text, [position[0], position[1], position[2] + 1.5], 
                       textColorRGB=[1, 0.5, 0], textSize=1.2)
    return fire_id

def add_beacon_marker(position, text="EMERGENCY"):
    """Add a yellow beacon marker"""
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], 
                                 rgbaColor=[1, 1, 0, 1])
    beacon_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, 
                                  basePosition=position)
    p.addUserDebugText(text, [position[0], position[1], position[2] + 1.5], 
                       textColorRGB=[1, 1, 0], textSize=1.2)
    return beacon_id

# ==============================================================================
# MAIN VISUALIZATION SCRIPT
# ==============================================================================
def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Add compass labels
    z_h = 0.5
    p.addUserDebugText("NORTH (+X)", [15, 0, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("SOUTH (-X)", [-10, 0, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("EAST (+Y)", [0, 15, z_h], [0, 0, 0], textSize=1.0)
    p.addUserDebugText("WEST (-Y)", [0, -15, z_h], [0, 0, 0], textSize=1.0)
    
    # Place emergency markers
    print("\n[System] Placing emergency markers...")
    
    # HUMAN IN THE MIDDLE (Center of environment)
    survivor_pos = [0, 0, 1.0]  # Changed to center
    add_human_marker(survivor_pos)
    print(f"[Scene] Added SURVIVOR at MIDDLE: {survivor_pos}")
    
    # Fire at east boundary
    fire_pos = [0, 40, 1.0]
    add_fire_marker(fire_pos)
    print(f"[Scene] Added FIRE at: {fire_pos}")
    
    # Emergency beacon at north-east boundary
    beacon_pos = [30, 30, 1.0]
    add_beacon_marker(beacon_pos, "BEACON")
    print(f"[Scene] Added BEACON at: {beacon_pos}")
    
    print("\n[System] Visualization ready. Close PyBullet window to exit.")
    
    # Keep simulation running
    while True:
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    main()