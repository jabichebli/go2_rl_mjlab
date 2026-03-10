#!/usr/bin/env python3
"""Show which poses are being used at different curriculum stages."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter",type=int, default=17500)
args = parser.parse_args()

# Curriculum parameters
START_ITER = 10000
FULL_ITER = 17500  
STEPS_PER_ITER = 24

def get_arm_scale(iteration):
    """Get arm curriculum scale at iteration."""
    current_step = iteration * STEPS_PER_ITER
    start_step = START_ITER * STEPS_PER_ITER
    full_step = FULL_ITER * STEPS_PER_ITER
    
    if current_step < start_step:
        return 0.0
    else:
        return min(1.0, (current_step - start_step) / (full_step - start_step))

# Pose names
POSE_NAMES = [
    "0: Tucked (safe)",
    "1-3: Full forward extended (L/C/R)",
    "4-6: Full backward extended (L/C/R)", 
    "7-8: Horizontal sweep (L/R)",
    "9-16: Arc positions (various J1 angles)",
    "17-19: Forward + wrist bent",
    "20-23: GROUND-REACHING (center/L/R/+wrist)",
    "24-27: Diagonal extremes + wrist rolls"
]

# Show curriculum timeline
print("=" * 70)
print(f"POSE USAGE AT ITERATION {args.iter}")
print("=" * 70)
print()

scale = get_arm_scale(args.iter)
print(f"Arm curriculum scale: {scale:.1%}")
print()

if scale == 0.0:
    print("ARM IS TUCKED - No extreme poses active")
    print("Robot is learning to walk with arm safely folded")
elif scale < 1.0:
    print(f"ARM IS RAMPING - Using mix of tucked ({(1-scale)*100:.0f}%) and extreme poses ({scale*100:.0f}%)")
    print()
    print("Active poses:")
    print("  - Tucked (baseline)")
    print("  - Forward/backward extended")
    print("  - Horizontal sweeps")  
    print("  - Arc positions")
    print("  - Ground-reaching")
    print("  - Diagonal extremes")
    print()
    print(f"Each extreme pose has {scale*100:.0f}% chance of being used vs tucked")
else:
    print("ARM IS AT 100% - MAXIMUM EXTREME SWEEPING!")
    print()
    print("ALL 28 POSES ACTIVE INCLUDING:")
    for name in POSE_NAMES:
        marker = " ⚠️" if "GROUND" in name or "extreme" in name.lower() else ""
        print(f"  • {name}{marker}")
    print()
    print("⚠️  Robot experiencing MAXIMUM destabilization from arm movements")

print()
print("=" * 70)
print("CURRICULUM TIMELINE:")
print("=" * 70)

key_iters = [0, 5000, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
for it in key_iters:
    arm_scale = get_arm_scale(it)
    
    # Determine phase
    if it < 10000:
        phase = "Walk only (arm tucked)"
    elif it < 17500:
        phase = f"Arm ramping ({arm_scale*100:.0f}%)"
    elif it < 22500:
        phase = "Arm 100%, weight ramping"
    else:
        phase = "Full arm + full weight"
    
    marker = " ← YOU ARE HERE" if it == args.iter else ""
    print(f"  {it:5d} | {phase:30s}{marker}")

print("=" * 70)
