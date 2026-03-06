"""
Visualization script for extreme_arm_sweep algorithm.

This runs the EXACT SAME algorithm as events.py:extreme_arm_sweep()
so you can visualize and verify it works before training.

Run with: python scripts/test_extreme_sweep.py
"""

import time
import math
import mujoco
import mujoco.viewer
import numpy as np

from mjlab.asset_zoo.robots.unitree_go2_arm.go2_arm_constants import get_go2_arm_robot_cfg
from mjlab.entity import Entity


# === EXACT SAME POSES AS events.py:extreme_arm_sweep() ===
# Format: [J0, J1, J2, J3, J4, J5, Gripper]
POSES = np.array([
    # 0: Tucked (safe neutral) - arm folded against body
    [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0],
    
    # === FULL FORWARD EXTENDED (J1=90°, J2=-90°) ===
    # Maximum forward reach - huge torque pulling robot forward
    # 1: Full forward left
    [-2.3, 1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 2: Full forward center  
    [0.0, 1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 3: Full forward right
    [2.3, 1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    
    # === FULL BACKWARD EXTENDED (J1=-90°, J2=-90°) ===
    # Maximum rear reach - huge torque pulling robot backward
    # 4: Full back left
    [-2.3, -1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 5: Full back center
    [0.0, -1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 6: Full back right
    [2.3, -1.571, -1.571, 0.0, 0.0, 0.0, 0.0],
    
    # === HORIZONTAL EXTENDED (J1=0°, J2=-90°) - shoulder height ===
    # Side-to-side torque at max lever arm
    # 7: Horizontal left
    [-2.3, 0.0, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 8: Horizontal right
    [2.3, 0.0, -1.571, 0.0, 0.0, 0.0, 0.0],
    
    # === ARC POSITIONS (various J1 with arm extended) ===
    # Intermediate angles for continuous coverage
    # 9: Arc 60° (J1=1.047) left
    [-2.3, 1.047, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 10: Arc 60° right
    [2.3, 1.047, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 11: Arc 30° (J1=0.524) left
    [-2.3, 0.524, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 12: Arc 30° right
    [2.3, 0.524, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 13: Arc -30° (J1=-0.524) left
    [-2.3, -0.524, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 14: Arc -30° right
    [2.3, -0.524, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 15: Arc -60° (J1=-1.047) left
    [-2.3, -1.047, -1.571, 0.0, 0.0, 0.0, 0.0],
    # 16: Arc -60° right
    [2.3, -1.047, -1.571, 0.0, 0.0, 0.0, 0.0],
    
    # === WRIST VARIATIONS (extended forward with wrist pitched) ===
    # 17: Forward extended + wrist bent down
    [0.0, 1.571, -1.571, 0.0, 1.571, 0.0, 0.0],
    # 18: Forward left + wrist bent
    [-2.3, 1.571, -1.571, 0.0, 1.571, 0.0, 0.0],
    # 19: Forward right + wrist bent
    [2.3, 1.571, -1.571, 0.0, 1.571, 0.0, 0.0],
    
    # === GROUND-REACHING (arm extended downward - maximum low torque) ===
    # J1=90° (forward), J2=0° (elbow partially bent down)
    # These reach toward ground but DON'T touch it
    # 20: Ground reach left
    [-2.3, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
    # 21: Ground reach right  
    [2.3, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
    # 22: Ground reach center-left (J0=-1.5, not full left to avoid body collision)
    [-1.5, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
    # 23: Ground reach center-right
    [1.5, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    # === DIAGONAL EXTREMES (from test_sweep.py) ===
    # These create complex off-axis torques
    # 24: Diagonal up-left (backward + rolled)
    [2.2, -1.571, 0.5, -2.3, -1.5, 2.3, 0.0],
    # 25: Diagonal up-right
    [-2.2, -1.571, 0.5, 2.3, -1.5, -2.3, 0.0],
    # 26: Diagonal down-left (forward + rolled)
    [2.2, 1.2, -0.5, -2.3, 1.5, 2.3, 0.0],
    # 27: Diagonal down-right
    [-2.2, 1.2, -0.5, 2.3, 1.5, -2.3, 0.0],
])

POSE_NAMES = [
    "Tucked",
    "Full Fwd Left", "Full Fwd Center", "Full Fwd Right",
    "Full Back Left", "Full Back Center", "Full Back Right", 
    "Horizontal Left", "Horizontal Right",
    "Arc 60° Left", "Arc 60° Right",
    "Arc 30° Left", "Arc 30° Right",
    "Arc -30° Left", "Arc -30° Right",
    "Arc -60° Left", "Arc -60° Right",
    "Fwd+Wrist Center", "Fwd+Wrist Left", "Fwd+Wrist Right",
    "Ground Left", "Ground Right", "Ground CenterL", "Ground CenterR",
    "Diag Up-Left", "Diag Up-Right", "Diag Down-Left", "Diag Down-Right",
]


def get_arm_target_random(step: int, env_id: int = 0, show_curriculum: bool = False):
    """
    EXACT SAME ALGORITHM as events.py:extreme_arm_sweep()
    
    This creates pseudo-random pose transitions that are:
    - Different for each env_id
    - Smooth (cosine interpolation)
    - Unpredictable (model can't learn timing)
    - VARIABLE SPEED per environment and per segment
    - Speed also RAMPS with curriculum (slow -> faster)
    
    Args:
        step: Current simulation step
        env_id: Environment ID (different IDs = different sequences AND speeds)
        show_curriculum: If True, apply curriculum scaling based on step
    """
    num_poses = len(POSES)
    
    # === CURRICULUM (same as events.py) ===
    if show_curriculum:
        START_ITER = 7500
        FULL_ITER = 12500
        STEPS_PER_ITER = 24
        start_step_curr = START_ITER * STEPS_PER_ITER   # 180,000 steps
        full_step_curr = FULL_ITER * STEPS_PER_ITER     # 300,000 steps
        
        if step < start_step_curr:
            scale = 0.0
        else:
            scale = min(1.0, (step - start_step_curr) / (full_step_curr - start_step_curr))
    else:
        scale = 1.0  # No curriculum - show full movement
    
    # === VARIABLE SPEED (same as events.py) ===
    # REALISTIC SPEEDS for robot arm:
    #   - Slowest: 2500 steps (50 seconds) - very deliberate movement
    #   - Fastest: 1000 steps (20 seconds) - quick but achievable
    #
    # CURRICULUM affects speed: starts slow-only, gradually allows faster
    SLOWEST_TRANSITION = 2500  # 50 seconds
    FASTEST_TRANSITION = 1000  # 20 seconds
    
    # Min speed depends on curriculum scale (slow -> fast as training progresses)
    MIN_TRANSITION_STEPS = int(SLOWEST_TRANSITION - (SLOWEST_TRANSITION - FASTEST_TRANSITION) * scale)
    MAX_TRANSITION_STEPS = SLOWEST_TRANSITION
    
    # Deterministic per-env base speed
    speed_hash = ((env_id * 7127 + 2731) % 1000) / 1000.0
    base_transition_steps = int(MIN_TRANSITION_STEPS + (MAX_TRANSITION_STEPS - MIN_TRANSITION_STEPS) * speed_hash)
    
    # === DESYNCHRONIZATION (same algorithm) ===
    time_offset = (env_id * 7919) % (MAX_TRANSITION_STEPS * num_poses)
    adjusted_step = step + time_offset
    
    # Segment tracking (use average for boundaries)
    AVG_TRANSITION = (MIN_TRANSITION_STEPS + MAX_TRANSITION_STEPS) // 2
    segment_idx = adjusted_step // AVG_TRANSITION
    
    # Per-segment speed variation (0.85 to 1.15x - less extreme)
    segment_speed_hash = ((env_id * 6271 + segment_idx * 3571) % 1000) / 1000.0
    segment_speed_factor = 0.85 + segment_speed_hash * 0.30
    
    # Effective transition for THIS segment
    effective_transition = int(base_transition_steps / segment_speed_factor)
    effective_transition = max(1000, min(2500, effective_transition))
    
    # Progress within current transition (0.0 to 1.0)
    progress = (adjusted_step % AVG_TRANSITION) / float(effective_transition)
    progress = max(0.0, min(1.0, progress))
    
    # Smooth interpolation (same cosine ease as events.py and test_sweep.py)
    smooth_progress = (1.0 - math.cos(progress * math.pi)) / 2.0
    
    # === PSEUDO-RANDOM POSE SELECTION (EXACT SAME HASH as events.py) ===
    current_pose_idx = (env_id * 6271 + segment_idx * 7919 + env_id * segment_idx * 127) % num_poses
    next_pose_idx = (env_id * 6271 + (segment_idx + 1) * 7919 + env_id * (segment_idx + 1) * 127) % num_poses
    
    current_pose = POSES[current_pose_idx]
    next_pose = POSES[next_pose_idx]
    
    # Interpolate
    target = current_pose + (next_pose - current_pose) * smooth_progress
    
    # Apply curriculum - blend toward tucked if scale < 1
    if scale < 1.0:
        tucked = POSES[0]
        target = tucked + (target - tucked) * scale
    
    # Return transition time in seconds for display
    transition_seconds = effective_transition / 50.0  # Assuming 50Hz
    
    return target[:6], current_pose_idx, next_pose_idx, smooth_progress, scale, transition_seconds


def run_random_visualization(env_id: int = 0):
    """
    Visualize the random arm sweep algorithm for a specific env_id.
    
    Args:
        env_id: Which environment's trajectory to visualize (0-4095 are all different)
    """
    # Get base speed for this env_id
    speed_hash = ((env_id * 7127 + 2731) % 1000) / 1000.0
    base_speed_seconds = (100 + (300 - 100) * speed_hash) / 50.0
    
    print(f"Loading Go2 model...")
    print(f"Visualizing env_id={env_id}")
    print(f"Base transition speed: {base_speed_seconds:.1f}s (varies ±40% per segment)")
    print(f"Total poses: {len(POSES)}")
    print("-" * 60)
    
    robot = Entity(get_go2_arm_robot_cfg())
    
    robot.spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, 
        size=[5, 5, 0.1],
        rgba=[0.3, 0.3, 0.3, 1]
    )
    
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    arm_ids = []
    leg_ids = {}
    
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not actuator_name: 
            continue
            
        if any(f"joint{j}" in actuator_name for j in range(1, 7)):
            arm_ids.append(i)
        elif "hip" in actuator_name:   
            leg_ids[i] = 0.0     
        elif "thigh" in actuator_name: 
            leg_ids[i] = 0.8     
        elif "calf" in actuator_name:  
            leg_ids[i] = -1.5    

    print("Launching viewer... (Close the window to stop)")
    print("=" * 60)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        dt = model.opt.timestep
        last_current_idx = -1
        
        while viewer.is_running():
            step_start = time.time()

            # Lock the legs
            for act_id, target_pos in leg_ids.items():
                data.ctrl[act_id] = target_pos

            # Get arm targets using the SAME ALGORITHM as training
            arm_targets, current_idx, next_idx, progress, scale, trans_time = get_arm_target_random(step, env_id)
            
            # Print when transitioning to a new pose
            if current_idx != last_current_idx:
                print(f"Step {step:6d}: [{POSE_NAMES[current_idx]:18s}] -> [{POSE_NAMES[next_idx]:18s}] ({trans_time:.1f}s)")
                last_current_idx = current_idx

            # Apply to arm motors
            for i, act_id in enumerate(arm_ids):
                if i < 6: 
                    data.ctrl[act_id] = arm_targets[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            
            step += 1
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def run_curriculum_demo():
    """
    Demonstrates the CURRICULUM learning phases by fast-forwarding through training.
    Shows how arm movement scales from 0% to 100%.
    """
    print("=" * 70)
    print("CURRICULUM LEARNING DEMONSTRATION")
    print("=" * 70)
    print("\nPhase 1: 0 -> 7,500 iterations     = Arm TUCKED (learn to walk)")
    print("Phase 2: 7,500 -> 12,500 iterations = Arm scales 0% -> 100%")
    print("Phase 3: 12,500+ iterations         = Weight added (disabled for now)")
    print("\n" + "-" * 70)
    
    # Sample steps at key points
    test_iterations = [0, 2500, 5000, 7500, 8750, 10000, 11250, 12500, 15000]
    STEPS_PER_ITER = 24
    
    for iter_num in test_iterations:
        step = iter_num * STEPS_PER_ITER
        _, _, _, _, scale, _ = get_arm_target_random(step, env_id=0, show_curriculum=True)
        
        phase = "Phase 1 (TUCKED)" if iter_num < 7500 else \
                "Phase 2 (RAMPING)" if iter_num < 12500 else \
                "Phase 3 (FULL)"
        
        bar_len = int(scale * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"Iter {iter_num:6d} | Scale: {scale:.2f} [{bar}] | {phase}")
    
    print("\n" + "=" * 70)


def compare_multiple_envs():
    """
    Show that different env_ids produce different sequences AND speeds.
    Prints the first 10 pose transitions for env_ids 0, 1, 2, 3.
    """
    print("=" * 70)
    print("COMPARING POSE SEQUENCES AND SPEEDS FOR DIFFERENT ENV_IDS")
    print("Each parallel environment sees different sequences at different speeds")
    print("=" * 70)
    
    for env_id in [0, 1, 2, 3, 100, 500]:
        speed_hash = ((env_id * 7127 + 2731) % 1000) / 1000.0
        base_speed = (100 + (300 - 100) * speed_hash) / 50.0
        print(f"\nEnv {env_id} (base speed: {base_speed:.1f}s) - first 10 transitions:")
        for segment in range(10):
            step = segment * 200  # Approximate segment start
            _, current_idx, next_idx, _, _, trans_time = get_arm_target_random(step, env_id)
            print(f"  {POSE_NAMES[current_idx]:18s} -> {POSE_NAMES[next_idx]:18s} ({trans_time:.1f}s)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_multiple_envs()
        elif sys.argv[1] == "--curriculum":
            run_curriculum_demo()
        else:
            # Run visualization with specific env_id
            env_id = int(sys.argv[1])
            run_random_visualization(env_id)
    else:
        # Show curriculum demo and comparison first
        run_curriculum_demo()
        print()
        compare_multiple_envs()
        print("\nStarting visualization for env_id=0...")
        print("Run with: python test_extreme_sweep.py <env_id> to see different sequences")
        print("Run with: python test_extreme_sweep.py --compare to just see sequences")
        print("Run with: python test_extreme_sweep.py --curriculum to see curriculum phases\n")
        run_random_visualization(env_id=0)
