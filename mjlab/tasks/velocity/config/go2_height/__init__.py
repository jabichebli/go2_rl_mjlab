from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_height_flat_env_cfg,   # EDITED
  unitree_go2_height_rough_env_cfg,  # EDITED
)
from .rl_cfg import unitree_go2_height_ppo_runner_cfg # EDITED

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go2-Height", # EDITED
  env_cfg=unitree_go2_height_rough_env_cfg(),
  play_env_cfg=unitree_go2_height_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2-Height", # EDITED
  env_cfg=unitree_go2_height_flat_env_cfg(),
  play_env_cfg=unitree_go2_height_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)