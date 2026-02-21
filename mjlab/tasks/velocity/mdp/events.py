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
  low = asset.data.joint_limits[asset_cfg.joint_ids, 0]
  high = asset.data.joint_limits[asset_cfg.joint_ids, 1]
  
  # Sample new random positions
  num_resets = len(env_ids)
  num_joints = len(asset_cfg.joint_ids)
  new_targets = low + torch.rand((num_resets, num_joints), device=env.device) * (high - low)
  
  # Write the targets directly to the simulator's control registers
  # This moves the arm without the RL policy knowing or caring
  actuator_ids = asset_cfg.actuator_ids
  env.sim.data.ctrl[env_ids[:, None], actuator_ids] = new_targets