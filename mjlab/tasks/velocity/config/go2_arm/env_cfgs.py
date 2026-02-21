"""Unitree Go2 with D1 Arm velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  get_go2_arm_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg  # ADDED
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_arm_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 + Arm rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # 1. Use the robot with the arm
  cfg.scene.entities = {"robot": get_go2_arm_robot_cfg()}

  # Helper variables for grouping
  leg_joints = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
  arm_joints = ["joint[1-6]"] # D1 Arm joints

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # 2. SENSORS (Keeping your logic, which includes arm collision safety)
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
      pattern=r".*_collision\d*$",
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

  # 3. ACTIONS (Strictly Leg Control to protect the walking gait)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.joint_names = leg_joints

  # 4. OBSERVATIONS (Feeling the arm weight/position)
  # Fix: Use SceneEntityCfg so the simulator knows which joints to look at
  cfg.observations["policy"].terms["arm_joint_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)}
  )
  cfg.observations["policy"].terms["arm_joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)}
  )

  # 5. EVENTS (Training robustness by moving the arm during training)
  cfg.events["randomize_arm_reset"] = EventTermCfg(
      func=mdp.randomize_joint_targets, 
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)}
  )
  cfg.events["randomize_arm_interval"] = EventTermCfg(
      func=mdp.randomize_joint_targets,
      mode="interval",
      interval_range_s=(4.0, 10.0),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)}
  )

  # 6. VIEWER & BASE CONFIG
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # 7. REWARDS (Identical to your original file)
  # CRITICAL: We tell the pose reward to ONLY look at leg joints. 
  # This prevents the arm's movement from messing up the walking reward score.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=leg_joints)

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

  # All other walking rewards remain untouched
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # 8. PLAY MODE OVERRIDES
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("randomize_arm_interval", None)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_arm_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 + Arm flat terrain velocity configuration."""
  cfg = unitree_go2_arm_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  del cfg.curriculum["terrain_levels"]

  return cfg