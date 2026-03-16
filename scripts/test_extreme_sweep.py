"""
Visualization script for extreme_arm_sweep algorithm.

This mirrors the pose library and the main timing behavior from
events.py:extreme_arm_sweep() closely enough to debug the sweep set by eye.

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
    # 0: Tucked (safe neutral) - arm folded back against body
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
    
    # === GROUND-REACHING (forward/downward without passing through the torso) ===
    # Keep some elbow extension so the arm stays in front of the body during
    # transitions instead of folding through the torso volume.
    # 20: Ground reach center
    [0.0, 1.25, -0.85, 0.0, 0.0, 0.0, 0.0],
    # 21: Ground reach left
    [-2.3, 1.25, -0.85, 0.0, 0.0, 0.0, 0.0],
    # 22: Ground reach right
    [2.3, 1.25, -0.85, 0.0, 0.0, 0.0, 0.0],
    # 23: Ground reach center with wrist
    [0.0, 1.25, -0.85, 0.0, 1.2, 0.0, 0.0],
    
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
    "Ground Center", "Ground Left", "Ground Right", "Ground+Wrist",
    "Diag Up-Left", "Diag Up-Right", "Diag Down-Left", "Diag Down-Right",
]


def get_arm_target_random(
    step: int,
    env_id: int = 0,
    show_curriculum: bool = False,
    random_offset: int | None = None,
    override_iter: int | None = None,
    play_speed_override: float | None = None,
):
    """
    EXACT SAME ALGORITHM as events.py:extreme_arm_sweep()
    
    This creates pseudo-random pose transitions that are:
    - Different for each env_id
    - Smooth (cosine interpolation)
    - Unpredictable (model can't learn timing)
    - CONTINUOUS motion (no holding - always moving!)
    
    Args:
        step: Current simulation step
        env_id: Environment ID (different IDs = different sequences)
        show_curriculum: If True, apply curriculum scaling based on step
        random_offset: Random time offset (simulates play mode randomness)
        override_iter: If set, pretend training is at this iteration (for --iter mode)
    """
    num_poses = len(POSES)
    
    # === CURRICULUM (same as events.py) ===
    if show_curriculum or override_iter is not None:
        START_ITER = 5000
        FULL_ITER = 20000
        STEPS_PER_ITER = 24
        
        if override_iter is not None:
            # Use the override iteration to calculate scale
            effective_step = override_iter * STEPS_PER_ITER
        else:
            effective_step = step
        
        start_step_curr = START_ITER * STEPS_PER_ITER
        full_step_curr = FULL_ITER * STEPS_PER_ITER
        
        if effective_step < start_step_curr:
            scale = 0.0
        else:
            scale = min(1.0, (effective_step - start_step_curr) / (full_step_curr - start_step_curr))
    else:
        scale = 1.0  # No curriculum - show full movement
    
    # === SPEED CURRICULUM / PLAY OVERRIDE ===
    # Default behavior approximates the current training logic:
    # - each env gets a persistent speed profile in [4, 12]
    # - that profile ramps with curriculum progress
    # For direct play comparisons, pass play_speed_override to use a fixed speed.
    if play_speed_override is not None:
        effective_speed_multiplier = float(play_speed_override)
    else:
        min_speed_multiplier = 4.0
        max_speed_multiplier = 12.0
        env_hash = ((env_id * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF
        env_speed_multiplier = min_speed_multiplier + (
            max_speed_multiplier - min_speed_multiplier
        ) * env_hash
        effective_speed_multiplier = env_speed_multiplier * (0.5 + 0.5 * scale)
    segment_duration = max(1, int(2500 / effective_speed_multiplier))
    
    # === DESYNCHRONIZATION ===
    if random_offset is not None:
        time_offset = random_offset
    else:
        time_offset = (env_id * 7919) % (2500 * num_poses)
    adjusted_step = step + time_offset
    
    # Segment tracking
    segment_idx = adjusted_step // segment_duration
    
    # Progress within segment - ALWAYS 0→1 over full segment (continuous motion!)
    steps_in_segment = adjusted_step % segment_duration
    progress = steps_in_segment / float(segment_duration)
    
    # Smooth interpolation (same cosine ease as events.py)
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
    
    # Return transition time in seconds for display.
    transition_seconds = segment_duration / 50.0

    return (
        target[:6],
        current_pose_idx,
        next_pose_idx,
        smooth_progress,
        scale,
        transition_seconds,
        effective_speed_multiplier,
    )


def run_random_visualization(
    env_id: int = 0,
    at_iter: int | None = None,
    play_speed_override: float | None = None,
):
    """
    Visualize the random arm sweep algorithm for a specific env_id.
    
    Args:
        env_id: Which environment's trajectory to visualize (0-4095 are all different)
        at_iter: If set, simulate as if training is at this iteration (shows curriculum effect)
    """
    # Use a random offset to simulate how play mode now works
    import random
    random_offset = random.randint(0, 2500 * len(POSES))
    
    print(f"Loading Go2 model...")
    print(f"Visualizing env_id={env_id}")
    if at_iter is not None:
        print(f"Simulating training at iteration {at_iter}")
    if play_speed_override is not None:
        print(f"Using exact play-style speed override: x{play_speed_override:.2f}")
    print(f"Random time offset: {random_offset} (different each launch, like play mode)")
    if play_speed_override is None:
        print("Transition speed: curriculum-dependent training profile")
    else:
        print("Transition speed: fixed play-style override")
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
        step = 0  # Policy step counter (increments every decimation sim steps)
        sim_step = 0  # Raw sim step counter
        dt = model.opt.timestep
        decimation = 4  # Match the env config: policy runs at 50Hz (200Hz sim / 4)
        last_current_idx = -1
        
        while viewer.is_running():
            step_start = time.time()

            # Lock the legs
            for act_id, target_pos in leg_ids.items():
                data.ctrl[act_id] = target_pos

            # Get arm targets using the SAME ALGORITHM as training
            # Only recalculate every decimation steps (matches policy rate)
            arm_targets, current_idx, next_idx, progress, scale, trans_time, speed_mult = get_arm_target_random(
                step, env_id, random_offset=random_offset,
                override_iter=at_iter,
                play_speed_override=play_speed_override,
            )
            
            # Print when transitioning to a new pose
            if current_idx != last_current_idx:
                scale_str = f" [scale={scale:.2f}]" if at_iter is not None else ""
                print(
                    f"Step {step:6d}: [{POSE_NAMES[current_idx]:18s}] -> "
                    f"[{POSE_NAMES[next_idx]:18s}] "
                    f"({trans_time:.1f}s, x{speed_mult:.2f}){scale_str}"
                )
                last_current_idx = current_idx

            # Apply to arm motors
            for i, act_id in enumerate(arm_ids):
                if i < 6: 
                    data.ctrl[act_id] = arm_targets[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            
            sim_step += 1
            # Only increment policy step every decimation sim steps
            if sim_step % decimation == 0:
                step += 1
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def run_curriculum_demo():
    """
    Demonstrates the CURRICULUM learning phases by fast-forwarding through training.
    Shows how arm movement and weight scale over iterations.
    """
    print("=" * 70)
    print("CURRICULUM LEARNING DEMONSTRATION")
    print("=" * 70)
    print()
    print("Phase 1:      0 -> 5,000 iters   = Arm TUCKED (learn to walk)")
    print("Phase 2:  5,000 -> 20,000 iters  = Arm ramps 0% -> 100%")
    print("Phase 3: 17,500 -> 22,500 iters  = Weight ramps 0% -> 100% (sequential!)")
    print()
    print("Tuck pose: [J0=0, J1=-1.5, J2=1.5, ...] = arm folded back against body")
    print()
    print("-" * 70)
    
    # Sample steps at key points
    test_iterations = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
    STEPS_PER_ITER = 24
    
    # Weight curriculum params (from randomize_payload_mass)
    WEIGHT_START = 17500
    WEIGHT_FULL = 22500
    
    # Arm curriculum params
    ARM_START = 5000
    ARM_FULL = 20000
    
    for iter_num in test_iterations:
        step = iter_num * STEPS_PER_ITER
        _, _, _, _, arm_scale, _, _ = get_arm_target_random(step, env_id=0, show_curriculum=True)
        
        # Weight scale
        if iter_num < WEIGHT_START:
            weight_scale = 0.0
        else:
            weight_scale = min(1.0, (iter_num - WEIGHT_START) / (WEIGHT_FULL - WEIGHT_START))
        
        if iter_num < ARM_START:
            phase = "WALK ONLY (tucked)"
        elif iter_num < ARM_FULL:
            phase = "ARM RAMP"
        elif iter_num < WEIGHT_START:
            phase = "FULL ARM (no weight)"
        elif iter_num < WEIGHT_FULL:
            phase = "FULL ARM + WEIGHT RAMP"
        else:
            phase = "FULL EVERYTHING"
        
        arm_bar = int(arm_scale * 15)
        weight_bar = int(weight_scale * 15)
        print(f"Iter {iter_num:6d} | Arm: {arm_scale:.2f} [{'█' * arm_bar}{'░' * (15 - arm_bar)}]"
              f" | Weight: {weight_scale:.2f} [{'█' * weight_bar}{'░' * (15 - weight_bar)}]"
              f" | {phase}")
    
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
        _, _, _, _, _, trans_time0, speed_mult0 = get_arm_target_random(0, env_id)
        print(f"\nEnv {env_id} (approx pose time: {trans_time0:.1f}s, x{speed_mult0:.2f}) - first 10 transitions:")
        for segment in range(10):
            step = segment * 200  # Approximate segment start
            _, current_idx, next_idx, _, _, trans_time, speed_mult = get_arm_target_random(step, env_id)
            print(f"  {POSE_NAMES[current_idx]:18s} -> {POSE_NAMES[next_idx]:18s} ({trans_time:.1f}s, x{speed_mult:.2f})")
    
    print("\n" + "=" * 70)


def run_walk_through():
    """
    Auto-advances through curriculum phases in the MuJoCo viewer.
    Shows the arm at key training iteration points, pausing briefly at each.
    """
    import random
    print("=" * 70)
    print("WALK-THROUGH: Showing arm at each curriculum phase")
    print("=" * 70)
    
    phases = [
        (0,     "Phase 1: TUCKED (learning to walk)"),
        (5000,  "Phase 1: TUCKED (still learning to walk)"),
        (5000,  "Phase 2 START: Arm begins moving (scale=0%)"),
        (10000, "Phase 2: Arm at 33% scale"),
        (15000, "Phase 2: Arm at 67% scale"),
        (20000, "Phase 2 DONE: Full arm"),
        (22500, "Phase 3: Full arm + Weight 100%"),
    ]
    
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
    
    random_offset = random.randint(0, 2500 * len(POSES))
    
    phase_idx = 0
    current_phase_iter, current_phase_label = phases[phase_idx]
    SECONDS_PER_PHASE = 15  # Show each phase for 15 seconds
    phase_steps = 0
    
    print(f"\n>>> {current_phase_label}")
    print(f"    (showing for {SECONDS_PER_PHASE}s, then advancing...)\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        dt = model.opt.timestep
        last_current_idx = -1
        
        while viewer.is_running():
            step_start = time.time()
            
            for act_id, target_pos in leg_ids.items():
                data.ctrl[act_id] = target_pos
            
            arm_targets, current_idx, next_idx, progress, scale, trans_time, speed_mult = get_arm_target_random(
                step, env_id=0, random_offset=random_offset,
                override_iter=current_phase_iter,
            )
            
            if current_idx != last_current_idx:
                print(
                    f"  Step {step:6d}: [{POSE_NAMES[current_idx]:18s}] -> "
                    f"[{POSE_NAMES[next_idx]:18s}] (scale={scale:.2f}, "
                    f"{trans_time:.1f}s, x{speed_mult:.2f})"
                )
                last_current_idx = current_idx
            
            for i, act_id in enumerate(arm_ids):
                if i < 6:
                    data.ctrl[act_id] = arm_targets[i]
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step += 1
            phase_steps += 1
            
            # Auto-advance after SECONDS_PER_PHASE
            if phase_steps >= int(SECONDS_PER_PHASE / dt) and phase_idx < len(phases) - 1:
                phase_idx += 1
                current_phase_iter, current_phase_label = phases[phase_idx]
                phase_steps = 0
                last_current_idx = -1
                print(f"\n>>> {current_phase_label}")
                print(f"    (showing for {SECONDS_PER_PHASE}s...)\n")
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print("Usage: python test_extreme_sweep.py [MODE]")
        print()
        print("Modes:")
        print("  <env_id>          Visualize arm sweep for specific env_id (e.g., 0, 1, 42)")
        print("  --iter <N>        Visualize arm at training iteration N (see curriculum effect)")
        print("                    e.g., --iter 0 = TUCK, --iter 10000 = 50% arm, --iter 15000 = full")
        print("  --play-speed <S>  Visualize with exact fixed play speed (e.g., --play-speed 4)")
        print("  --walk-through    Auto-advance through all curriculum phases in viewer")
        print("  --curriculum      Print curriculum timeline (text only)")
        print("  --compare         Compare pose sequences for different env_ids (text only)")
        print("  (no args)         Show curriculum text + compare + launch viewer")
        sys.exit(0)
    
    if "--walk-through" in args:
        run_walk_through()
    elif "--iter" in args:
        idx = args.index("--iter")
        if idx + 1 < len(args):
            iter_num = int(args[idx + 1])
        else:
            print("Error: --iter requires an iteration number (e.g., --iter 10000)")
            sys.exit(1)
        print(f"Showing arm behavior at training iteration {iter_num}")
        run_random_visualization(env_id=0, at_iter=iter_num)
    elif "--play-speed" in args:
        idx = args.index("--play-speed")
        if idx + 1 < len(args):
            speed = float(args[idx + 1])
        else:
            print("Error: --play-speed requires a numeric multiplier (e.g., --play-speed 4)")
            sys.exit(1)
        run_random_visualization(env_id=0, play_speed_override=speed)
    elif "--compare" in args:
        compare_multiple_envs()
    elif "--curriculum" in args:
        run_curriculum_demo()
    elif len(args) == 1 and args[0].isdigit():
        env_id = int(args[0])
        run_random_visualization(env_id)
    else:
        # Default: show curriculum text + compare + launch viewer
        run_curriculum_demo()
        print()
        compare_multiple_envs()
        print("\nStarting visualization for env_id=0...")
        print("Usage:")
        print("  python test_extreme_sweep.py <env_id>          # Specific env")
        print("  python test_extreme_sweep.py --iter <N>        # At specific iteration")
        print("  python test_extreme_sweep.py --iter 0          # See TUCK pose")
        print("  python test_extreme_sweep.py --iter 10000      # See 50% arm ramp")
        print("  python test_extreme_sweep.py --iter 15000      # See full arm + weight")
        print("  python test_extreme_sweep.py --walk-through    # Auto-advance phases")
        print("  python test_extreme_sweep.py --curriculum      # Text curriculum diagram")
        print("  python test_extreme_sweep.py --compare         # Compare env sequences")
        print()
        run_random_visualization(env_id=0)
