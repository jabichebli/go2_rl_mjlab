from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_arm_flat_env_cfg,   # CHANGED: Import arm-specific versions
  unitree_go2_arm_rough_env_cfg,  # CHANGED: Import arm-specific versions
)
from .rl_cfg import unitree_go2_arm_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go2-Arm", # CHANGED: Unique ID
  env_cfg=unitree_go2_arm_rough_env_cfg(),
  play_env_cfg=unitree_go2_arm_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_arm_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2-Arm",  # CHANGED: Unique ID
  env_cfg=unitree_go2_arm_flat_env_cfg(),
  play_env_cfg=unitree_go2_arm_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_arm_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)