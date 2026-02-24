"""Unitree Go2 with D1 Arm constants.

NOTE ON D1 ARM INERTIA AND CONTROL TUNING:
The <inertial> tags in a URDF define the link inertias (the physical mass 
and weight distribution of the metal/plastic arm pieces). However, the 
EFFECTIVE_INERTIAS used in the mjlab actuator configurations represent the 
reflected rotor inertia of the motors (calculated as J_eff = J_rotor * N^2, 
where N is the gear ratio). In highly geared robotic arms, the motor's 
reflected inertia drastically dominates the physical link inertia.

Since Unitree doesn't publicly publish the exact J_rotor and gear ratios 
for the D1 motors, the EFFECTIVE_INERTIAS below are reverse-calculated. 
By targeting a natural frequency of 4 Hz (standard for this size arm) 
and critical damping (zeta = 1.0), the effective inertias were derived to 
perfectly output the Kp ~= 20 stiffness known to safely hold the arm 
up against gravity.
"""

import math
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

# --- GO2 DOG ACTUATORS (Retained hardcoded leg values) ---
GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

# --- D1 ARM ACTUATORS (Physics-based tuning via YAM protocol) ---
D1_NATURAL_FREQ = 4.0 * 2 * math.pi
D1_DAMPING_RATIO = 1.0

# Reverse-derived to achieve Kp~20 for the base and Kp~15 for wrists
D1_EFFECTIVE_INERTIAS = {
  "joint1": 0.0316,
  "joint2": 0.0316,
  "joint3": 0.0237,
  "joint4": 0.0237,
  "joint5": 0.0237,
  "joint6": 0.0237,
  "left_finger": 0.0316,
}

D1_EFFORT_LIMITS = {
  "joint1": 3.3,
  "joint2": 3.3,
  "joint3": 1.7,
  "joint4": 1.7,
  "joint5": 1.7,
  "joint6": 1.7,
  "left_finger": 1.7,
}

D1_ACTUATORS = [
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=D1_EFFECTIVE_INERTIAS[name] * D1_NATURAL_FREQ**2,
    damping=2.0 * D1_DAMPING_RATIO * D1_EFFECTIVE_INERTIAS[name] * D1_NATURAL_FREQ,
    effort_limit=D1_EFFORT_LIMITS[name],
    armature=0.01,
  )
  for name in D1_EFFECTIVE_INERTIAS.keys()
]


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
    # D1 Arm Folded Pose
    "joint1": 0.0,
    "joint2": 1.5,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

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
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
    *D1_ACTUATORS,
  ),
  soft_joint_pos_limit_factor=0.95,
)

def get_go2_arm_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 + D1 arm robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARM_ARTICULATION,
  )

##
# Action Scaling (YAM Protocol)
##

GO2_ARM_ACTION_SCALE: dict[str, float] = {}

for a in GO2_ARM_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_ARM_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_arm_robot_cfg())
  
  # Print the generated action scales for verification before launch
  print("--- Generated Action Scales ---")
  for joint, scale in GO2_ARM_ACTION_SCALE.items():
      print(f"{joint}: {scale:.4f} rads")
  print("-------------------------------")

  viewer.launch(robot.spec.compile())
