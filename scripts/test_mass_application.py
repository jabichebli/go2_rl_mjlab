#!/usr/bin/env python3
"""Live visualization of mass application on link_6 during curriculum."""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=20000, help="Simulate this iteration")
parser.add_argument("--mass", type=float, help="Override with specific mass in grams")
args = parser.parse_args()

# Load model
model = mujoco.MjModel.from_xml_path("mjlab/asset_zoo/robots/unitree_go2_arm/xmls/go2_arm.xml")
data = mujoco.MjData(model)

# Find link_6
link6_id = None
for i in range(model.nbody):
    if model.body(i).name == "link_6":
        link6_id = i
        break

if link6_id is None:
    print("ERROR: link_6 not found!")
    exit(1)

default_mass = model.body_mass[link6_id]

# Calculate mass at iteration
START_ITER = 17500
FULL_ITER = 22500
STEPS_PER_ITER = 24
MASS_RANGE = (0.0, 0.1)

if args.mass is not None:
    # User override
    payload = args.mass / 1000.0  # Convert g to kg
    new_mass = default_mass + payload
else:
    # Calculate from curriculum
    current_step = args.iter * STEPS_PER_ITER
    start_step = START_ITER * STEPS_PER_ITER
    full_step = FULL_ITER * STEPS_PER_ITER
    
    if current_step < start_step:
        scale = 0.0
    else:
        scale = min(1.0, (current_step - start_step) / (full_step - start_step))
    
    max_payload = MASS_RANGE[0] + (MASS_RANGE[1] - MASS_RANGE[0]) * scale
    
    # Use max payload for visualization
    payload = max_payload
    new_mass = default_mass + payload

# Apply the mass
model.body_mass[link6_id] = new_mass

print("=" * 60)
print("MASS APPLICATION TEST")
print("=" * 60)
print(f"Iteration:      {args.iter}")
print(f"link_6 body ID: {link6_id}")
print(f"Default mass:   {default_mass:.6f} kg ({default_mass*1000:.2f}g)")
print(f"Payload added:  {payload:.6f} kg ({payload*1000:.2f}g)")
print(f"Total mass:     {new_mass:.6f} kg ({new_mass*1000:.2f}g)")
print(f"Mass increase:  {((new_mass/default_mass - 1)*100):.1f}%")
print()
print(f"Torque @ 0.5m extension: {payload * 9.81 * 0.5:.3f} Nm")
print("=" * 60)
print()
print("VIEWER INSTRUCTIONS:")
print("  - link_6 is the last link before gripper fingers")
print("  - Watch body's inertia when arm moves")
print("  - Press TAB to see contact forces")
print("  - Press ESC to exit")
print("=" * 60)

# Set arm to forward extended position to show effect
# Joint ordering: J0, J1, J2, J3, J4, J5, gripper
forward_pose = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # Maximum reach forward

# Find joint IDs
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint_left_finger"]
for i, name in enumerate(joint_names):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id >= 0:
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr] = forward_pose[i]

# Forward kinematics to update positions
mujoco.mj_forward(model, data)

# Launch viewer with continuous stepping
def controller(model, data):
    """Simple controller to hold arm in extended position."""
    # Find actuator IDs and set targets to hold pose
    for i, name in enumerate(joint_names[:-1]):  # Skip gripper
        actuator_name = f"actuator_{name}"
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if act_id >= 0:
            data.ctrl[act_id] = forward_pose[i]

# Run viewer
print("\nLaunching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 45
    
    while viewer.is_running():
        step_start = time.time()
        
        # Apply control
        controller(model, data)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        viewer.sync()
        
        # Simple timing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
