import time
import math
import mujoco
import mujoco.viewer

from mjlab.asset_zoo.robots.unitree_go2_arm.go2_arm_constants import get_go2_arm_robot_cfg
from mjlab.entity import Entity

def get_arm_target(t):
    """
    EXTREME SWEEP TEST for RL Training - Body Stabilization Against Arm Movements
    
    Joint mapping (user's j1-j6 = our J0-J5):
    - J0 (Base Yaw) - sweeps left/right
    - J1 (Shoulder Pitch) - forward(+90°)/backward(-90°) 
    - J2 (Elbow Pitch) - straight(-90°)/bent(+90°)
    - J3 (Wrist Roll) - rotation
    - J4 (Wrist Pitch) - wrist up/down
    - J5 (Wrist Yaw) - wrist rotation
    
    Sequence:
    1. Full forward extension sweep (J0 sweep, arm straight)
    2. Full forward with wrist bent sweep (J0 sweep, wrist pitched)
    3. Full forward reaching ground sweep (J0 sweep, elbow+wrist bent)
    4. Full backward extension sweep
    5. Other extreme motions
    
    Joint limits: J0,J3,J5: ±135° (±2.356) | J1,J2,J4: ±90° (±1.571)
    """
    cycle_duration = 127.0
    t = t % cycle_duration

    # [J0 (Base Yaw), J1 (Shoulder Pitch), J2 (Elbow Pitch), 
    #  J3 (Wrist Roll), J4 (Wrist Pitch), J5 (Wrist Yaw)]
    
    tucked              = [ 0.0,   -1.5,    1.5,   0.0,    0.0,    0.0]

    # === SEQUENCE 1: FULL FORWARD EXTENDED (J0=0, J1=90°, J2=-90°) ===
    full_fwd_center     = [ 0.0,    1.571, -1.571,  0.0,    0.0,    0.0]
    full_fwd_left       = [-2.3,    1.571, -1.571,  0.0,    0.0,    0.0]  # J0 sweep left
    full_fwd_right      = [ 2.3,    1.571, -1.571,  0.0,    0.0,    0.0]  # J0 sweep right

    # === SEQUENCE 1B: FORWARD→BACKWARD ARC SWEEP (J1 transitions 90° to -90°) ===
    arc_60_left         = [-2.3,    1.047, -1.571,  0.0,    0.0,    0.0]  # J1 = 60°
    arc_60_right        = [ 2.3,    1.047, -1.571,  0.0,    0.0,    0.0]
    
    arc_30_left         = [-2.3,    0.524, -1.571,  0.0,    0.0,    0.0]  # J1 = 30°
    arc_30_right        = [ 2.3,    0.524, -1.571,  0.0,    0.0,    0.0]
    
    arc_0_left          = [-2.3,    0.0,   -1.571,  0.0,    0.0,    0.0]  # J1 = 0° (horizontal)
    arc_0_right         = [ 2.3,    0.0,   -1.571,  0.0,    0.0,    0.0]
    
    arc_neg30_left      = [-2.3,   -0.524, -1.571,  0.0,    0.0,    0.0]  # J1 = -30°
    arc_neg30_right     = [ 2.3,   -0.524, -1.571,  0.0,    0.0,    0.0]
    
    arc_neg60_left      = [-2.3,   -1.047, -1.571,  0.0,    0.0,    0.0]  # J1 = -60°
    arc_neg60_right     = [ 2.3,   -1.047, -1.571,  0.0,    0.0,    0.0]
    
    # === SEQUENCE 2: FULL FORWARD WITH WRIST BENT (J0 sweep, J4=90°) ===
    fwd_wrist_center    = [ 0.0,    1.571, -1.571,  0.0,    1.571,  0.0]
    fwd_wrist_left      = [-2.3,    1.571, -1.571,  0.0,    1.571,  0.0]
    fwd_wrist_right     = [ 2.3,    1.571, -1.571,  0.0,    1.571,  0.0]
    
    # === SEQUENCE 3: FULL FORWARD REACHING GROUND (J0 sweep, J2=90°, J4=90°) ===
    fwd_ground_center   = [ 0.0,   1.571,  -1.571,  0.0,    0.0,  0.0]  # Lift above head
    fwd_ground_left     = [-2.3,    1.571,  0.0,  0.0,    0.0,  0.0]
    fwd_ground_right    = [ 2.3,    1.571,  0.0,  0.0,    0.0,  0.0]
    
    # === SEQUENCE 4: FULL BACKWARD EXTENDED (J0=0, J1=-90°, J2=-90°) ===
    full_back_center    = [ 0.0,   -1.571, -1.571,  0.0,    0.0,    0.0]
    full_back_left      = [-2.3,   -1.571, -1.571,  0.0,    0.0,    0.0]
    full_back_right     = [ 2.3,   -1.571, -1.571,  0.0,    0.0,    0.0]
    
    # === OTHER EXTREME MOTIONS ===
    # Diagonal extremes
    diag_up_L           = [ 2.2,   -1.571,  0.5,  -2.3,   -1.5,    2.3]
    diag_up_R           = [-2.2,   -1.571,  0.5,   2.3,   -1.5,   -2.3]
    diag_down_L         = [ 2.2,    1.2,   -0.5,  -2.3,    1.5,    2.3]
    diag_down_R         = [-2.2,    1.2,   -0.5,   2.3,    1.5,   -2.3]
    
    # Mid-height sweeps
    mid_center          = [ 0.0,    0.0,    0.0,   0.0,    0.0,    0.0]
    mid_sweep_L         = [ 2.2,    0.0,    0.0,  -2.3,    0.0,    2.3]
    mid_sweep_R         = [-2.2,    0.0,    0.0,   2.3,    0.0,   -2.3]
    
    # Wrist extreme tests
    twist_1             = [ 0.0,    0.0,    0.0,   2.3,    1.5,    2.3]
    twist_2             = [ 0.0,    0.0,    0.0,  -2.3,   -1.5,   -2.3]

    # Smooth cosine interpolation
    def interp(pose_a, pose_b, progress):
        smooth_p = (1 - math.cos(progress * math.pi)) / 2
        return [a + (b - a) * smooth_p for a, b in zip(pose_a, pose_b)]

    # ========== CHOREOGRAPHY ==========
    
    # SEQUENCE 1: FULL FORWARD EXTENDED - J0 sweep (0-15s)
    if t < 2.0:    return interp(tucked, full_fwd_center, t / 2.0)
    elif t < 5.0:  return interp(full_fwd_center, full_fwd_left, (t - 2.0) / 3.0)
    elif t < 10.0: return interp(full_fwd_left, full_fwd_right, (t - 5.0) / 5.0)
    elif t < 13.0: return interp(full_fwd_right, full_fwd_left, (t - 10.0) / 3.0)  # Back sweep
    elif t < 15.0: return interp(full_fwd_left, full_fwd_center, (t - 13.0) / 2.0)
    
    # SEQUENCE 1B: FORWARD→BACKWARD ARC SWEEP - J1 transitions from 90° to -90° (15-45s)
    # J1 = 60° sweep
    elif t < 17.0: return interp(full_fwd_center, arc_60_left, (t - 15.0) / 2.0)
    elif t < 21.0: return interp(arc_60_left, arc_60_right, (t - 17.0) / 4.0)
    
    # J1 = 30° sweep
    elif t < 23.0: return interp(arc_60_right, arc_30_right, (t - 21.0) / 2.0)
    elif t < 27.0: return interp(arc_30_right, arc_30_left, (t - 23.0) / 4.0)
    
    # J1 = 0° (horizontal) sweep
    elif t < 29.0: return interp(arc_30_left, arc_0_left, (t - 27.0) / 2.0)
    elif t < 33.0: return interp(arc_0_left, arc_0_right, (t - 29.0) / 4.0)
    
    # J1 = -30° sweep
    elif t < 35.0: return interp(arc_0_right, arc_neg30_right, (t - 33.0) / 2.0)
    elif t < 39.0: return interp(arc_neg30_right, arc_neg30_left, (t - 35.0) / 4.0)
    
    # J1 = -60° sweep
    elif t < 41.0: return interp(arc_neg30_left, arc_neg60_left, (t - 39.0) / 2.0)
    elif t < 45.0: return interp(arc_neg60_left, arc_neg60_right, (t - 41.0) / 4.0)
    
    # SEQUENCE 2: FULL FORWARD WITH WRIST BENT - J0 sweep (45-57s)
    elif t < 47.0: return interp(arc_neg60_right, fwd_wrist_center, (t - 45.0) / 2.0)
    elif t < 50.0: return interp(fwd_wrist_center, fwd_wrist_left, (t - 47.0) / 3.0)
    elif t < 54.0: return interp(fwd_wrist_left, fwd_wrist_right, (t - 50.0) / 4.0)
    elif t < 57.0: return interp(fwd_wrist_right, fwd_wrist_center, (t - 54.0) / 3.0)
    
    # SEQUENCE 3: FULL FORWARD REACHING GROUND - J0 sweep (57-71s)
    elif t < 59.0: return interp(fwd_wrist_center, fwd_ground_center, (t - 57.0) / 2.0)
    elif t < 61.0: return interp(fwd_ground_center, fwd_ground_right, (t - 59.0) / 2.0)
    elif t < 63.0: return interp(fwd_ground_right, fwd_ground_center, (t - 61.0) / 2.0)  # Back to center
    elif t < 65.0: return interp(fwd_ground_center, fwd_ground_left, (t - 63.0) / 2.0)
    elif t < 67.0: return interp(fwd_ground_left, fwd_ground_center, (t - 65.0) / 2.0)  # Back to center
    elif t < 69.0: return interp(fwd_ground_center, fwd_ground_right, (t - 67.0) / 2.0)  # One more
    elif t < 71.0: return interp(fwd_ground_right, fwd_ground_center, (t - 69.0) / 2.0)
    
    # Tuck before going backward
    elif t < 73.0: return interp(fwd_ground_center, tucked, (t - 71.0) / 2.0)
    
    # SEQUENCE 4: FULL BACKWARD EXTENDED - J0 sweep (73-88s)
    elif t < 76.0: return interp(tucked, full_back_center, (t - 73.0) / 3.0)
    elif t < 79.0: return interp(full_back_center, full_back_left, (t - 76.0) / 3.0)
    elif t < 84.0: return interp(full_back_left, full_back_right, (t - 79.0) / 5.0)
    elif t < 88.0: return interp(full_back_right, full_back_center, (t - 84.0) / 4.0)
    
    # OTHER EXTREME MOTIONS (88-127s)
    # Diagonal extremes
    elif t < 90.0: return interp(full_back_center, diag_up_L, (t - 88.0) / 2.0)
    elif t < 93.0: return interp(diag_up_L, diag_up_R, (t - 90.0) / 3.0)
    elif t < 96.0: return interp(diag_up_R, diag_down_L, (t - 93.0) / 3.0)
    elif t < 99.0: return interp(diag_down_L, diag_down_R, (t - 96.0) / 3.0)
    
    # Mid-height sweeps
    elif t < 101.0: return interp(diag_down_R, mid_center, (t - 99.0) / 2.0)
    elif t < 104.0: return interp(mid_center, mid_sweep_L, (t - 101.0) / 3.0)
    elif t < 108.0: return interp(mid_sweep_L, mid_sweep_R, (t - 104.0) / 4.0)
    elif t < 111.0: return interp(mid_sweep_R, mid_sweep_L, (t - 108.0) / 3.0)
    elif t < 113.0: return interp(mid_sweep_L, mid_center, (t - 111.0) / 2.0)
    
    # Wrist extremes
    elif t < 115.0: return interp(mid_center, twist_1, (t - 113.0) / 2.0)
    elif t < 118.0: return interp(twist_1, twist_2, (t - 115.0) / 3.0)
    elif t < 121.0: return interp(twist_2, twist_1, (t - 118.0) / 3.0)
    elif t < 123.0: return interp(twist_1, mid_center, (t - 121.0) / 2.0)
    
    # Return home
    elif t < 127.0: return interp(mid_center, tucked, (t - 123.0) / 4.0)
    else:          return tucked

def run_isolated_sweep():
    print("Loading Go2 model with motors...")
    robot = Entity(get_go2_arm_robot_cfg())
    
    robot.spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, 
        size=[5, 5, 0.1],
        rgba=[0.3, 0.3, 0.3, 1]
    )
    
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    arm_ids = []
    leg_ids = {}
    
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not actuator_name: continue
            
        if any(f"joint{j}" in actuator_name for j in range(1, 7)):
            arm_ids.append(i)
        elif "hip" in actuator_name:   leg_ids[i] = 0.0     
        elif "thigh" in actuator_name: leg_ids[i] = 0.8     
        elif "calf" in actuator_name:  leg_ids[i] = -1.5    

    print("Launching viewer... (Close the window to stop)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        dt = model.opt.timestep
        
        while viewer.is_running():
            step_start = time.time()

            # 1. Lock the legs
            for act_id, target_pos in leg_ids.items():
                data.ctrl[act_id] = target_pos

            # 2. Get max-reach arm targets
            arm_targets = get_arm_target(t)

            # 3. Apply to arm motors
            for i, act_id in enumerate(arm_ids):
                if i < 6: 
                    data.ctrl[act_id] = arm_targets[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            
            t += dt
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_isolated_sweep()