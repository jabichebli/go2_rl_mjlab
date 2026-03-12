import torch
import math
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def extreme_arm_sweep(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg,
    scale_override: float | None = None,
    speed_multiplier: float = 1.0
):
    """
    Creates EXTREME, RANDOM arm movements using coordinated poses.
    
    KEY DIFFERENCES from simple sine waves:
    1. Uses coordinated multi-joint POSES (not independent joint oscillation)
    2. Poses are designed to create MAXIMUM torque on the body
    3. Each environment follows a DIFFERENT pseudo-random sequence
    4. Model cannot predict what comes next - forces reactive balancing
    
    Joint mapping:
    - J0 (idx 0): Base Yaw - sweeps left/right (±2.35 rad = ±135°)
    - J1 (idx 1): Shoulder Pitch - forward(+90°)/backward(-90°)
    - J2 (idx 2): Elbow Pitch - straight(-90°)/bent(+90°)
    - J3 (idx 3): Wrist Roll
    - J4 (idx 4): Wrist Pitch
    - J5 (idx 5): Wrist Yaw
    - Gripper (idx 6): Finger
    
    Args:
        scale_override: If provided, bypasses curriculum and uses this scale directly.
                       Set to 1.0 in play mode to see full arm range.
        speed_multiplier: Multiplier for arm movement speed. Default 1.0 for training.
                         Set to 4.0 in play mode for faster, more visible movements.
    """
    
    # === CURRICULUM LEARNING ===
    # === STAGGERED CURRICULUM ===
    # Key insight: NOT all environments should transition at the same time!
    # This prevents catastrophic forgetting of walking behavior.
    #
    # Phase 1: 0 -> 5k iterations = arm stays TUCKED (learn to walk first!)
    # Phase 2: 5k -> 20k iterations = GRADUAL ramp with per-env staggering
    #   - 20% of envs: ALWAYS scale=0 (pure walking reference)
    #   - 80% of envs: Random offset into curriculum (diversity)
    # 
    # This ensures the policy always sees some "easy walking" examples
    # and doesn't learn to freeze to survive arm movements.
    
    current_step = env.common_step_counter
    STEPS_PER_ITER = 24
    
    # Curriculum timing (in iterations)
    START_ITER = 5000      # Start arm movement earlier
    FULL_ITER = 20000      # Reach full scale later (more time to adapt)
    
    start_step = START_ITER * STEPS_PER_ITER
    full_step = FULL_ITER * STEPS_PER_ITER
    
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    num_envs = len(env_ids)
    
    # Calculate base scale from curriculum
    if scale_override is not None:
        # Play mode: use override for all envs
        scale = torch.full((num_envs,), scale_override, device=env.device)
    elif current_step < start_step:
        scale = torch.zeros(num_envs, device=env.device)
    else:
        base_progress = min(1.0, (current_step - start_step) / (full_step - start_step))
        
        # === PER-ENVIRONMENT STAGGERING ===
        # Create persistent random offsets for each environment
        if not hasattr(env, '_arm_curriculum_offsets'):
            # 20% of envs get scale=0 (pure walking), 80% get curriculum
            env._arm_curriculum_offsets = torch.rand(env.num_envs, device=env.device)
            # Mark 20% as "walking only" (offset = -1 means always scale=0)
            walking_only_mask = torch.rand(env.num_envs, device=env.device) < 0.2
            env._arm_curriculum_offsets[walking_only_mask] = -1.0
        
        offsets = env._arm_curriculum_offsets[env_ids]
        
        # Walking-only envs stay at scale=0
        # Others get staggered progress: their "personal" curriculum is offset
        # This means some envs are "ahead" and some are "behind" in the curriculum
        scale = torch.zeros(num_envs, device=env.device)
        active_mask = offsets >= 0
        
        # Staggered scale: base_progress adjusted by per-env offset
        # offset in [0, 1] shifts when this env "starts" its curriculum
        # env with offset=0.5 starts at 50% progress when base is 0.5, reaches 1.0 when base is 1.0
        staggered_progress = (base_progress - offsets[active_mask] * 0.5).clamp(0, 1)
        scale[active_mask] = staggered_progress
    
    # === EXTREME POSES (from test_sweep.py - these create maximum torque!) ===
    # Format: [J0, J1, J2, J3, J4, J5, Gripper]
    # These are the poses that make the robot work hardest to balance
    
    poses = torch.tensor([
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
        
        # === GROUND-REACHING (arm extended downward toward floor) ===
        # J1=90° forward, J2=0° (elbow straight down)
        # 20: Ground reach center
        [0.0, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
        # 21: Ground reach left
        [-2.3, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
        # 22: Ground reach right
        [2.3, 1.571, 0.0, 0.0, 0.0, 0.0, 0.0],
        # 23: Ground reach center with wrist
        [0.0, 1.571, 0.0, 0.0, 1.571, 0.0, 0.0],
        
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
        
    ], device=env.device)
    
    num_poses = poses.shape[0]
    
    # === VARIABLE SPEED TRANSITION ===
    # Each environment gets a DIFFERENT transition speed to:
    # 1. Prevent policy from learning timing
    # 2. Train robustness to fast AND slow arm movements
    # 3. Better match real-world arm capabilities
    #
    # REALISTIC SPEEDS for a robot arm:
    #   - Slow (3500 steps / 70s): Very careful, deliberate movements
    #   - Medium (2500 steps / 50s): Normal operation  
    #   - Fast (1750 steps / 35s): Quick but achievable
    #
    # SPEED CURRICULUM: Start SLOW, gradually allow faster movements
    # Early training: 2x slower (3500 steps per transition)
    # Late training: normal speed (1750 steps per transition)
    
    # Calculate speed factor from curriculum progress
    # At start of Phase 2 (scale near 0): speed_factor = 0.5 (2x slower)
    # At end of curriculum (scale = 1): speed_factor = 1.0 (normal)
    avg_scale = scale.mean().item() if scale.numel() > 0 else 0.0
    speed_factor = 0.5 + 0.5 * avg_scale  # Ramps from 0.5 to 1.0
    
    # Base duration (at speed_factor=1.0, speed_multiplier=1.0): 2500 steps = 50s per pose
    # During early training: 2500/0.5 = 5000 steps = 100s per pose (very slow!)
    # During play (speed_multiplier=4.0): 2500/4 = 625 steps = 12.5s per pose
    BASE_SEGMENT_DURATION = 2500
    SEGMENT_DURATION = int(BASE_SEGMENT_DURATION / (speed_multiplier * speed_factor))
    
    # === PERSISTENT RANDOM STATE ===
    # Created once per session. Ensures:
    # 1. Different play sessions see different arm sequences (not deterministic on env_id)
    # 2. Training still has full diversity across environments
    # 3. Reproducible within a session (same random seed → same offsets)
    if not hasattr(env, '_arm_sweep_state'):
        env._arm_sweep_state = {
            'time_offsets': torch.randint(
                0, SEGMENT_DURATION * num_poses,
                (env.num_envs,), device=env.device
            ),
        }
    
    # === DESYNCHRONIZATION ===
    # Random offsets ensure each env AND each play session gets a unique sequence
    time_offsets = env._arm_sweep_state['time_offsets'][env_ids]
    adjusted_steps = current_step + time_offsets
    
    # Which transition segment we're in
    segment_idx = adjusted_steps // SEGMENT_DURATION
    
    # Progress within current segment - ALWAYS 0→1 over the full segment (continuous motion!)
    steps_in_segment = adjusted_steps % SEGMENT_DURATION
    progress = steps_in_segment.float() / float(SEGMENT_DURATION)
    
    # Smooth interpolation using cosine ease-in-out
    smooth_progress = (1.0 - torch.cos(progress * math.pi)) / 2.0
    
    # === PSEUDO-RANDOM POSE SELECTION ===
    # Creates a deterministic but "random-looking" sequence per environment
    # Each env_id generates a different sequence of poses
    # Large primes create good mixing/distribution
    
    # Hash function: combines env_id and segment to pick pose
    current_pose_idx = ((env_ids * 6271 + segment_idx * 7919 + env_ids * segment_idx * 127) % num_poses).long()
    next_pose_idx = ((env_ids * 6271 + (segment_idx + 1) * 7919 + env_ids * (segment_idx + 1) * 127) % num_poses).long()
    
    # Gather poses for each environment
    current_poses = poses[current_pose_idx]  # Shape: [num_envs, 7]
    next_poses = poses[next_pose_idx]        # Shape: [num_envs, 7]
    
    # Interpolate between current and next pose
    targets = current_poses + (next_poses - current_poses) * smooth_progress.unsqueeze(1)
    
    # === APPLY CURRICULUM ===
    # scale is now per-environment tensor [num_envs]
    # Blend towards tucked pose based on each env's scale
    tucked = poses[0].unsqueeze(0).expand(num_envs, -1)
    # scale[:, None] broadcasts to [num_envs, 7]
    targets = tucked + (targets - tucked) * scale.unsqueeze(1)
    
    # FIX: Write to joint_pos_target instead of ctrl!
    # The ctrl gets overwritten by write_data_to_sim() which reads from joint_pos_target.
    # We must set joint_pos_target for the arm joints so apply_controls() uses our values.
    joint_ids = asset_cfg.joint_ids
    asset.data.joint_pos_target[env_ids[:, None], joint_ids] = targets


def randomize_joint_targets(
  env: ManagerBasedRlEnv, 
  env_ids: torch.Tensor, 
  asset_cfg: SceneEntityCfg
):
  """Randomizes control targets for specific joints (the arm)."""
  asset = env.scene[asset_cfg.name]
  
  # Get the joint limits for the arm joints defined in the config
  soft_limit_factor = 1.0  
  low = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 0] * soft_limit_factor
  high = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 1] * soft_limit_factor
  
  # Sample new random positions
  num_resets = len(env_ids)
  num_joints = len(asset_cfg.joint_ids)
  new_targets = low + torch.rand((num_resets, num_joints), device=env.device) * (high - low)
  
  # FIX: Write to joint_pos_target so it doesn't get overwritten by apply_controls()
  joint_ids = asset_cfg.joint_ids
  asset.data.joint_pos_target[env_ids[:, None], joint_ids] = new_targets


def smooth_arm_movement(
  env: ManagerBasedRlEnv, 
  env_ids: torch.Tensor, 
  asset_cfg: SceneEntityCfg
):
    """Actively holds the arm folded, then slowly sweeps it using gentle sine waves."""
    
    current_step = env.common_step_counter
    
    # 1. TIMING FIX: Match your 5k -> 10k iteration curriculum
    # Assuming 24 steps_per_env per iteration
    start_step = 120000  # 5,000 iterations 
    full_step = 240000   # 10,000 iterations 
    
    # scale = min(1.0, max(0.0, (current_step - start_step) / (full_step - start_step)))
    scale = 1.0
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    
    base_pose = torch.tensor([0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0], device=env.device)
    
    # 2. IF WE ARE WAITING: Actively hold the arm tight
    if scale <= 0.0:
        new_targets = base_pose.unsqueeze(0).repeat(len(env_ids), 1)
        # FIX: Write to joint_pos_target so it doesn't get overwritten
        asset.data.joint_pos_target[env_ids[:, None], joint_ids] = new_targets
        return 
        
    # 3. IF WE ARE MOVING: Calculate a gentle sweep
    # Tightened safety buffer to 0.8 to heavily prevent self-collision
    soft_limit_factor = 0.8  
    low = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 0] * soft_limit_factor
    high = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 1] * soft_limit_factor
    
    half_range = (high - low) / 2.0
    
    # Max amplitude is 30% of the safe range to keep the CoM shift manageable
    amplitude = half_range * 0.3 * scale 
    
    num_joints = len(asset_cfg.joint_ids)
    t = current_step * 0.02 # Assuming standard 50Hz control frequency
    
    # 4. FREQUENCY FIX: Lowered frequencies (0.05 to 0.15 Hz) 
    # This means a full swing takes about 6 to 20 seconds, preventing jerkiness.
    freqs = torch.linspace(0.5, 1.0, num_joints, device=env.device)
    joint_phases = torch.linspace(0, math.pi, num_joints, device=env.device)
                                  
    
    # Desynchronize the robots
    env_phase_offsets = (env_ids.unsqueeze(1) * 2.345) % (2 * math.pi)
    
    # Calculate the smooth target wave
    wave = torch.sin(2 * math.pi * freqs * t + joint_phases + env_phase_offsets)
    
    # Add the wave to the folded base pose
    new_targets = base_pose + amplitude * wave
    
    # FIX: Write to joint_pos_target so it doesn't get overwritten
    asset.data.joint_pos_target[env_ids[:, None], joint_ids] = new_targets

 
def randomize_payload_mass(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg, 
    mass_range: tuple[float, float] = (0.0, 0.1)
):
    """
    Randomizes the mass of the arm's end-effector (link_6) using a curriculum.
    
    CURRICULUM (Phase 3 - OVERLAPS with arm movement!):
    - 0 -> 10,000 iterations: NO weight (early arm learning)
    - 10,000 -> 15,000 iterations: Weight ramps 0% -> 100%
    
    This overlaps with arm phase (7.5k-12.5k) so the robot learns:
    - 7.5k-10k: Arm movement only (easy)
    - 10k-12.5k: Arm ramping + weight starting (combined challenge)
    - 12.5k-15k: Full arm + weight ramping
    """
    
    # --- CURRICULUM LOGIC (Phase 3: overlaps with arm!) ---
    current_step = env.common_step_counter
    
    START_ITER = 17500   # Start weight AFTER arm is fully ramped
    FULL_ITER = 22500    # Full weight range by this point
    STEPS_PER_ITER = 24
    
    start_step = START_ITER * STEPS_PER_ITER  # 240,000 steps
    full_step = FULL_ITER * STEPS_PER_ITER    # 360,000 steps
    
    # Calculate scale: 0.0 before start, ramps to 1.0 at full
    if current_step < start_step:
        scale = 0.0
    else:
        scale = min(1.0, (current_step - start_step) / (full_step - start_step))

    # 1. Safely get the pre-calculated body ID directly from the config
    body_id = asset_cfg.body_ids[0]
    
    # Exact mass from go2_arm.xml for link_6
    default_link_mass = 0.077892  
    
    # IF WE ARE WAITING: Just use the empty hand (default mass)
    if scale <= 0.0:
        env.sim.model.body_mass[env_ids, body_id] = default_link_mass
        return
        
    # IF WE ARE READY: Calculate the scaled random payload
    # This gradually increases the maximum possible random weight up to the 500g limit
    max_payload = mass_range[0] + (mass_range[1] - mass_range[0]) * scale
    
    num_resets = len(env_ids)
    
    # Sample a random weight between 0.0 and the current curriculum maximum
    random_payloads = torch.rand(num_resets, device=env.device) * max_payload
    
    # Set absolute new masses
    new_masses = default_link_mass + random_payloads
    
    # Apply batched update to the simulator
    env.sim.model.body_mass[env_ids, body_id] = new_masses


def reset_arm_to_folded(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg
):
    """Sets the arm control targets to the folded pose on reset to prevent snapping."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    
    folded_targets = torch.tensor([0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0], device=env.device)
    
    num_resets = len(env_ids)
    # FIX: Write to joint_pos_target so it doesn't get overwritten
    asset.data.joint_pos_target[env_ids[:, None], joint_ids] = folded_targets.repeat(num_resets, 1)


# def smooth_randomize_arm_targets(
#     env: ManagerBasedRlEnv, 
#     env_ids: torch.Tensor, 
#     asset_cfg: SceneEntityCfg
# ):
#     """Moves the arm smoothly, with a curriculum that scales difficulty over time."""
#     asset = env.scene[asset_cfg.name]
#     soft_limit_factor = 0.95  # Keep the safety buffer!
    
#     # 1. --- CURRICULUM LOGIC ---
#     current_step = env.common_step_counter
    
#     start_step = 120000  # 5,000 iterations * 24 steps_per_env = 120,000 steps
#     full_step = 240000   # 10,000 iterations * 24 steps_per_env = 240,000 steps
    
#     # Calculate a scale from 0.0 to 1.0 based on the current training step
#     scale = min(1.0, max(0.0, (current_step - start_step) / (full_step - start_step)))
    
#     # If we are before start_step, don't move the arm at all
#     if scale <= 0.0:
#         return
        
#     # Scale the maximum allowed nudge
#     base_max_delta = 0.2 
#     max_delta = base_max_delta * scale
#     # ---------------------------

#     # 2. Get functional limits
#     low = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 0] * soft_limit_factor
#     high = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 1] * soft_limit_factor

#     # 3. Get CURRENT control targets
#     actuator_ids = asset_cfg.actuator_ids
#     current_targets = env.sim.data.ctrl[env_ids[:, None], actuator_ids]
    
#     # 4. Generate random nudges between -max_delta and +max_delta
#     num_resets = len(env_ids)
#     num_joints = len(asset_cfg.joint_ids)
#     deltas = (torch.rand((num_resets, num_joints), device=env.device) * 2.0 - 1.0) * max_delta
    
#     # 5. Apply the nudge and clamp
#     new_targets = current_targets + deltas
#     new_targets = torch.clamp(new_targets, low, high)
    
#     # 6. Write back
#     env.sim.data.ctrl[env_ids[:, None], actuator_ids] = new_targets