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
