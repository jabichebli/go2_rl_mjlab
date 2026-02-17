# Unitree RL: Advanced Posture Control & Arm Manipulation

> **Note:** This repository is a heavily modified fork of the official [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) base. 
> 
> While the original repository provides an excellent baseline for standard locomotion (walking, running, yaw control) using MuJoCo and Isaac Lab abstractions, **this fork extends the action space and reward structure to support dynamic posture adjustments and robotic arm manipulation.**

## 🌟 Custom Features Added in This Fork

Unlike standard quadruped policies that only accept planar velocity commands (v_x, v_y, omega_z), this repository introduces:

* **Dynamic Torso Posture Control:** The RL policy has been modified to accept and execute commands for base height (z_height), pitch, and roll. This allows the robot to lower its profile to crawl under obstacles or tilt its chassis dynamically while moving.
* **Integrated Robotic Arm Manipulation:** The MuJoCo XML/URDF files have been updated to include a top-mounted robotic arm. The observation and action spaces have been expanded to co-simulate and control the extra Degrees of Freedom (DoF) alongside the locomotion policy.
* **Custom Reward Functions:** Added specific penalty and tracking terms to stabilize the quadruped's gait while shifting its Center of Mass (CoM) due to the arm's movements and extreme torso tilts.

---

## ✳️ Base Overview (from Original Mjlab)

This project utilizes MuJoCo as its physics simulation backend. It combines Isaac Lab's proven API with best-in-class MuJoCo physics to provide lightweight, modular abstractions for RL robotics research and sim-to-real deployment. 

<div align="center">

| Simulation (MuJoCo) | Physical Deployment |
| :---: | :---: |
| *[Insert your custom GIF of the dog tilting/using arm in sim here]* | *[Insert your custom GIF of the real dog here (if applicable)]* |

</div>

## 📦 Installation and Configuration

Please refer to the original [setup.md](doc/setup_en.md) for the base environment installation and configuration steps.

## 🛠️ Usage Guide: Custom Policies

### 1. Training the Posture & Arm Policy

To train the custom policy that includes the extended posture controls (pitch, roll, height) and arm manipulation, run the following command:

    python scripts/train.py Mjlab-Posture-Arm-Unitree-Go2 --env.scene.num-envs=4096

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

    python scripts/train.py Mjlab-Posture-Arm-Unitree-Go2 \
      --gpu-ids 0 1 \
      --env.scene.num-envs=4096

### 2. Validating the Policy in Simulation

To visualize your trained posture and arm policy interacting in the MuJoCo viewer:

    python scripts/play.py Mjlab-Posture-Arm-Unitree-Go2 --checkpoint_file=logs/rsl_rl/go2_posture_arm/2026-xx-xx_xx-xx-xx/model_xx.pt

*(For standard velocity or mimic training instructions on unmodified robots, please refer to `ORIGINAL_README.md`)*

---

## 🔁 Sim-to-Real Deployment

The custom policies trained in this fork maintain compatibility with Unitree's Sim2Real pipeline. During training, `policy.onnx` and `policy.onnx.data` are exported. 

To deploy to a physical robot, you must ensure your real robot's SDK is configured to accept the extended action array (locomotion + arm joints).

1. Power on the robot in suspended state and enter `zero-torque` mode.
2. Press `L2 + R2` on the controller to enter `debug mode`.
3. Connect via Ethernet (`192.168.123.222`).
4. Place your exported `.onnx` files into your custom deploy configuration folder and compile using `CMake`.
5. Execute the compiled control script via the appropriate network interface.

*(See `ORIGINAL_README.md` for detailed compilation flags for standard Unitree robots).*

## 🎉 Acknowledgements

This extended project builds upon the incredible work of the open-source robotics community:

* **[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)**: The original baseline repository for this fork.
* **[mjlab](https://github.com/mujocolab/mjlab.git)**: Training and execution framework.
* **[rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)**: Reinforcement learning algorithm implementation.
* **[mujoco](https://github.com/google-deepmind/mujoco.git)**: High-fidelity rigid-body physics engine.
