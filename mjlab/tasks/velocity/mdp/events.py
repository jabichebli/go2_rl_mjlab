import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
  
  # Write the targets directly to the simulator's control registers
  # This moves the arm without the RL policy knowing or caring
  actuator_ids = asset_cfg.actuator_ids
  env.sim.data.ctrl[env_ids[:, None], actuator_ids] = new_targets
  
  
 
def randomize_payload_mass(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg, 
    mass_range: tuple[float, float] = (0.0, 0.5)
):
    """Randomizes the mass of the arm's end-effector (link_6)."""
    
    # 1. Safely get the pre-calculated body ID directly from the config
    body_id = asset_cfg.body_ids[0]

    # 2. Sample random payload masses (up to 500g)
    num_resets = len(env_ids)
    random_payloads = mass_range[0] + torch.rand(num_resets, device=env.device) * (mass_range[1] - mass_range[0])
    
    # 3. Use exact mass from go2_arm.xml for link_6
    default_link_mass = 0.077892  
    
    # 4. Set absolute new masses
    new_masses = default_link_mass + random_payloads
    
    # 5. Apply batched update
    env.sim.model.body_mass[env_ids, body_id] = new_masses


def reset_arm_to_folded(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg
):
    """Sets the arm control targets to the folded pose on reset to prevent snapping."""
    asset = env.scene[asset_cfg.name]
    actuator_ids = asset_cfg.actuator_ids
    
    folded_targets = torch.tensor([0.0, -1.5, 1.5, 0.0, 0.0, 0.0,0.0], device=env.device)
    
    num_resets = len(env_ids)
    env.sim.data.ctrl[env_ids[:, None], actuator_ids] = folded_targets.repeat(num_resets, 1)


def smooth_randomize_arm_targets(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg
):
    """Moves the arm smoothly, with a curriculum that scales difficulty over time."""
    asset = env.scene[asset_cfg.name]
    soft_limit_factor = 0.95  # Keep the safety buffer!
    
    # 1. --- CURRICULUM LOGIC ---
    current_step = env.common_step_counter
    
    start_step = 120000  # 5,000 iterations * 24 steps_per_env = 120,000 steps
    full_step = 240000   # 10,000 iterations * 24 steps_per_env = 240,000 steps
    
    # Calculate a scale from 0.0 to 1.0 based on the current training step
    scale = min(1.0, max(0.0, (current_step - start_step) / (full_step - start_step)))
    
    # If we are before start_step, don't move the arm at all
    if scale <= 0.0:
        return
        
    # Scale the maximum allowed nudge
    base_max_delta = 0.2 
    max_delta = base_max_delta * scale
    # ---------------------------

    # 2. Get functional limits
    low = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 0] * soft_limit_factor
    high = asset.data.joint_pos_limits[0, asset_cfg.joint_ids, 1] * soft_limit_factor

    # 3. Get CURRENT control targets
    actuator_ids = asset_cfg.actuator_ids
    current_targets = env.sim.data.ctrl[env_ids[:, None], actuator_ids]
    
    # 4. Generate random nudges between -max_delta and +max_delta
    num_resets = len(env_ids)
    num_joints = len(asset_cfg.joint_ids)
    deltas = (torch.rand((num_resets, num_joints), device=env.device) * 2.0 - 1.0) * max_delta
    
    # 5. Apply the nudge and clamp
    new_targets = current_targets + deltas
    new_targets = torch.clamp(new_targets, low, high)
    
    # 6. Write back
    env.sim.data.ctrl[env_ids[:, None], actuator_ids] = new_targets