"""Unitree Go2 with D1 Arm constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# CHANGED: Pointing to the new go2_arm.xml file
GO2_ARM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2_arm" / "xmls" / "go2_arm.xml"
)
assert GO2_ARM_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_ARM_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_ARM_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# --- GO2 DOG ACTUATORS ---
GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

# --- D1 ARM ACTUATORS (NEW) ---
# Joints 1 & 2 (Yaw and Pitch base) have stronger 3.3 NM motors
D1_ACTUATOR_STRONG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "joint1",
    "joint2",
  ),
  stiffness=5.0,  # Equivalent to kp=5
  damping=0.5,    # Equivalent to kv=0.5
  effort_limit=3.3, # 3.3 NM max torque
  armature=0.01,
)

# Joints 3-6 and the gripper have 1.7 NM motors
D1_ACTUATOR_WEAK = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "joint3",
    "joint4",
    "joint5",
    "joint6",
  ),
  stiffness=5.0,
  damping=0.5,
  effort_limit=1.7, # 1.7 NM max torque
  armature=0.01,
)

D1_ACTUATOR_CLAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_finger",
  ),
  stiffness=5.0,
  damping=0.5,
  effort_limit=1.7,
  armature=0.01,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.4),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
    # Arm default folded pose (Adjust these radians to match how the arm should rest)
    "joint1": 0.0,
    "joint2": 1.5,  # Bends forward
    "joint3": -1.5, # Folds back
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,

    # The arm joints will naturally default to 0.0, which means the arm points straight up.
    # If you want it to start folded, you can add them here (e.g., "joint2": 1.57)
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # Go2 actuators
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
    # D1 actuators
    D1_ACTUATOR_STRONG,
    D1_ACTUATOR_WEAK,
    D1_ACTUATOR_CLAW,
  ),
  soft_joint_pos_limit_factor=0.9, # RL safety margin
)


def get_go2_arm_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 + D1 arm robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARM_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  # Load the combined robot
  robot = Entity(get_go2_arm_robot_cfg())

  # Launch the viewer!
  viewer.launch(robot.spec.compile())