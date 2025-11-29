#!/usr/bin/env python3
"""
Test script to see exploration waypoint mechanics.
Creates dummy main waypoints and prints exploration sequence one by one.
"""

import time

# ==============================================================================
# DUMMY DATA
# ==============================================================================
DUMMY_MAIN_WAYPOINTS = [
    [5.0, 0.0, 1.5],
    [10.0, 0.0, 1.5],
    [15.0, 0.0, 1.5],
    [20.0, 0.0, 1.5],
    [25.0, 0.0, 1.5],
    [30.0, 0.0, 1.5],
    [35.0, 0.0, 1.5],
    [40.0, 0.0, 1.5],
    [45.0, 0.0, 1.5],
    [49.0, 0.0, 1.5],
]

# ==============================================================================
# EXPLORATION NAVIGATOR CLASS
# ==============================================================================
class ExplorationNavigator:
    """Navigator that generates exploration sequence for each main waypoint"""
    
    def __init__(self, direction="north", search_width=3.0):
        self.temp_direction = direction
        self.search_width = search_width
        self.current_main_idx = 0
        self.exploration_queue = []
        self.exploration_idx = 0
    
    def generate_exploration_sequence(self, main_waypoint):
        """
        Generate [left, center, right, center] pattern for a main waypoint
        
        Args:
            main_waypoint: [x, y, z] - the main LLM waypoint
        
        Returns:
            list: 4 exploration waypoints
        """
        if self.temp_direction in ["north", "south"]:
            left_wp = [main_waypoint[0], main_waypoint[1] - self.search_width, main_waypoint[2]]
            right_wp = [main_waypoint[0], main_waypoint[1] + self.search_width, main_waypoint[2]]
        else:
            left_wp = [main_waypoint[0] - self.search_width, main_waypoint[1], main_waypoint[2]]
            right_wp = [main_waypoint[0] + self.search_width, main_waypoint[1], main_waypoint[2]]
        
        return [left_wp, main_waypoint, right_wp, main_waypoint]
    
    def get_next_target(self):
        """Get next exploration point, returns None if all done"""
        if not self.exploration_queue:
            if self.current_main_idx >= len(DUMMY_MAIN_WAYPOINTS):
                return None
            
            main_wp = DUMMY_MAIN_WAYPOINTS[self.current_main_idx]
            self.exploration_queue = self.generate_exploration_sequence(main_wp)
            self.exploration_idx = 0
            print(f"\n{'='*60}")
            print(f"🔍 EXPLORING MAIN WP {self.current_main_idx + 1}/{len(DUMMY_MAIN_WAYPOINTS)}")
            print(f"Main WP: {main_wp}")
            print(f"Exploration pattern: [LEFT, CENTER, RIGHT, CENTER]")
            print(f"{'='*60}")
        
        target = self.exploration_queue[self.exploration_idx]
        exploration_names = ["LEFT", "CENTER", "RIGHT", "CENTER"]
        print(f"\n📍 Getting target: {exploration_names[self.exploration_idx]} → {target}")
        return target
    
    def advance(self):
        """Move to next exploration point. Returns True if more exist."""
        self.exploration_idx += 1
        
        if self.exploration_idx >= len(self.exploration_queue):
            print(f"\n✅ FINISHED Main WP {self.current_main_idx + 1}")
            self.current_main_idx += 1
            self.exploration_idx = 0
            self.exploration_queue = []
            
            if self.current_main_idx >= len(DUMMY_MAIN_WAYPOINTS):
                return False
        
        return True

# ==============================================================================
# TEST DRIVER
# ==============================================================================
def test_exploration_mechanics():
    """Main test that pulls waypoints one by one and prints everything"""
    print("=" * 70)
    print("TESTING EXPLORATION WAYPOINT MECHANICS")
    print("=" * 70)
    print(f"Direction: North (straight +X)")
    print(f"Main waypoint count: {len(DUMMY_MAIN_WAYPOINTS)}")
    print(f"Search width: ±3.0m (left/right)")
    print("Each main WP → 4 exploration steps")
    print("=" * 70)
    
    navigator = ExplorationNavigator(direction="north", search_width=3.0)
    step_count = 0
    
    print("\n🚁 Starting exploration sequence...")
    
    while True:
        # Get next target
        target = navigator.get_next_target()
        
        if target is None:
            print("\n" + "=" * 70)
            print("🎉 ALL WAYPOINTS FULLY EXPLORED!")
            print("=" * 70)
            break
        
        step_count += 1
        
        # Simulate drone "reaching" the point
        print(f"Step {step_count:2d} → Drone moving to: {target}")
        time.sleep(0.1)  # Simulate flight time
        
        # Check if point is reached (simulate with distance < 0.5)
        # In real code, this would be actual drone position check
        print(f"   ✓ Reached target: {target}")
        
        # Advance to next exploration point
        if not navigator.advance():
            break
    
    # Final statistics
    print("\n📊 EXPLORATION STATISTICS")
    print("=" * 70)
    print(f"Total exploration steps: {step_count}")
    print(f"Main waypoints explored: {len(DUMMY_MAIN_WAYPOINTS)}")
    print(f"Steps per main waypoint: 4")
    print(f"Expected total steps: {len(DUMMY_MAIN_WAYPOINTS) * 4}")
    print(f"Exploration efficiency: {(len(DUMMY_MAIN_WAYPOINTS) * 4) / step_count * 100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    test_exploration_mechanics()