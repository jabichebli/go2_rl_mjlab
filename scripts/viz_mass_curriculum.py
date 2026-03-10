#!/usr/bin/env python3
"""Visualize where payload mass is applied and how it changes over curriculum."""

import sys
import argparse
import mujoco
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize mass curriculum")
parser.add_argument("--model", type=str, default="mjlab/asset_zoo/robots/unitree_go2_arm/xmls/go2_arm.xml")
parser.add_argument("--iter", type=int, help="Show mass at specific iteration")
args = parser.parse_args()

# Load the model
try:
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("=" * 60)
print("PAYLOAD MASS VISUALIZATION")
print("=" * 60)
print()

# Find link_6 body
link6_id = None
for i in range(model.nbody):
    name = model.body(i).name
    if name == "link_6":
        link6_id = i
        break

if link6_id is None:
    print("ERROR: link_6 body not found!")
    sys.exit(1)

# Show arm chain
print("ARM BODY CHAIN:")
print("-" * 60)
for i in range(model.nbody):
    name = model.body(i).name
    if name and "link" in name.lower():
        mass = model.body_mass[i]
        marker = " ← PAYLOAD APPLIED HERE" if i == link6_id else ""
        print(f"  Body {i:2d}: {name:15s}  mass={mass:.6f} kg{marker}")
print()

# Show default mass
default_mass = model.body_mass[link6_id]
print(f"DEFAULT link_6 mass: {default_mass:.6f} kg ({default_mass*1000:.2f}g)")
print()

# Curriculum parameters
START_ITER = 17500
FULL_ITER = 22500
STEPS_PER_ITER = 24
MASS_RANGE = (0.0, 0.1)  # 0-100g

start_step = START_ITER * STEPS_PER_ITER
full_step = FULL_ITER * STEPS_PER_ITER

def get_mass_at_iter(iteration):
    """Calculate mass at given iteration."""
    current_step = iteration * STEPS_PER_ITER
    
    if current_step < start_step:
        scale = 0.0
    else:
        scale = min(1.0, (current_step - start_step) / (full_step - start_step))
    
    max_payload = MASS_RANGE[0] + (MASS_RANGE[1] - MASS_RANGE[0]) * scale
    
    # Return range (min, max) since it's random
    return (default_mass + MASS_RANGE[0], default_mass + max_payload)

print("MASS CURRICULUM:")
print("-" * 60)
print(f"Start iteration: {START_ITER}")
print(f"Full iteration:  {FULL_ITER}")
print(f"Payload range:   {MASS_RANGE[0]*1000:.0f}g - {MASS_RANGE[1]*1000:.0f}g")
print()

# Show curriculum stages
key_iters = [0, 5000, 10000, 15000, 17500, 20000, 22500, 25000]
print("Iteration | Curriculum | Min Mass | Max Mass | Max Payload")
print("-" * 60)
for it in key_iters:
    min_mass, max_mass = get_mass_at_iter(it)
    max_payload = max_mass - default_mass
    
    # Curriculum label
    if it < 10000:
        label = "Walk only"
    elif it < 17500:
        label = "Arm moving"
    elif it < 22500:
        label = "Weight ramp"
    else:
        label = "Full weight"
    
    print(f"{it:5d}     | {label:12s} | {min_mass:.6f} | {max_mass:.6f} | {max_payload*1000:4.0f}g")

print()

# If specific iteration requested
if args.iter is not None:
    print("=" * 60)
    print(f"AT ITERATION {args.iter}:")
    print("=" * 60)
    min_mass, max_mass = get_mass_at_iter(args.iter)
    max_payload = max_mass - default_mass
    
    print(f"link_6 mass range: {min_mass:.6f} kg - {max_mass:.6f} kg")
    print(f"                   ({min_mass*1000:.2f}g - {max_mass*1000:.2f}g)")
    print(f"Payload range:     0g - {max_payload*1000:.0f}g")
    print()
    
print("=" * 60)
print("NOTES:")
print("-" * 60)
print("• Mass is applied to link_6 (end-effector)")
print("• Each episode gets a RANDOM mass between 0 and curriculum max")
print("• Mass affects inertia at full arm extension (~0.5m)")
print(f"• Max torque @ full extension: {MASS_RANGE[1] * 9.81 * 0.5:.3f} Nm")
print("=" * 60)
