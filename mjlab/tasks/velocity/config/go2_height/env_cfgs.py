"""Unitree Go2 height tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg # ADDED
from mjlab.managers.reward_manager import RewardTermCfg         # ADDED
from mjlab.managers.scene_entity_config import SceneEntityCfg   # ADDED for height-adaptive foot clearance
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_height_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg: # EDITED
  """Create Unitree Go2 height tracking rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # --- ADDED: Height Command Configuration ---
  cfg.commands["twist"].ranges.base_height = (0.20, 0.40)

  # --- ADDED: Height Observations ---
  cfg.observations["policy"].terms["base_height"] = ObservationTermCfg(
      func=mdp.base_height
  )
  cfg.observations["policy"].terms["commanded_height"] = ObservationTermCfg(
      func=mdp.commanded_height,
      params={"command_name": "twist"}
  )

  # --- ADDED: Height Tracking Reward ---
  cfg.rewards["track_base_height"] = RewardTermCfg(
      func=mdp.track_base_height,
      weight=2.0,  
      params={"std": 0.05, "command_name": "twist"},
  )

  # --- Reduce pose reward weight to allow height adjustment ---
  if "pose" in cfg.rewards:
      cfg.rewards["pose"].weight *= 0.1 

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)
  
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # --- Pose std values (same as original go2, weight reduced above) ---
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  
  # --- Replace fixed foot_clearance with height-adaptive version ---
  del cfg.rewards["foot_clearance"]
  cfg.rewards["foot_clearance"] = RewardTermCfg(
      func=mdp.feet_clearance_height_adaptive,
      weight=-1.0,
      params={
          "target_height_at_max": 0.10,   # 10cm clearance at 0.40m body height (normal)
          "target_height_at_min": 0.03,   # 3cm clearance at 0.20m body height (crouched, legs have less room)
          "height_command_name": "twist",
          "height_range": (0.20, 0.40),
          "command_name": "twist",
          "command_threshold": 0.1,
          "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
  )
  
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_height_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg: # EDITED
  """Create Unitree Go2 height tracking flat terrain velocity configuration."""
  cfg = unitree_go2_height_rough_env_cfg(play=play) # EDITED

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  del cfg.curriculum["terrain_levels"]

  return cfg