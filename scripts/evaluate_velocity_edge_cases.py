"""Headless edge-case evaluation for velocity policies.

This script avoids the viewer entirely and instead measures how often a trained
policy stalls or under-tracks commanded motion, both overall and in a few
arm-pose-conditioned edge cases.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateEdgeConfig:
  checkpoint_file: str
  """Local checkpoint to evaluate."""

  num_envs: int = 64
  """Parallel environments to run."""

  num_steps: int = 2000
  """Environment steps to evaluate."""

  warmup_steps: int = 200
  """Initial steps to skip from metric accumulation."""

  device: str | None = None
  """Execution device. Defaults to CUDA when available."""

  play: bool = True
  """Use play-mode env config. This is usually what you want for stress testing."""

  output_file: str | None = None
  """Optional JSON output path."""


def _case_stats(
  counts: dict[str, float],
  name: str,
  mask: torch.Tensor,
  cmd_mag: torch.Tensor,
  actual_mag: torch.Tensor,
  projection: torch.Tensor,
) -> None:
  count = float(mask.sum().item())
  counts[f"{name}/count"] += count
  if count <= 0:
    return

  stalls = mask & (actual_mag < 0.10)
  undertrack = mask & (projection < 0.5 * cmd_mag)

  counts[f"{name}/stall_count"] += float(stalls.sum().item())
  counts[f"{name}/undertrack_count"] += float(undertrack.sum().item())
  counts[f"{name}/projection_sum"] += float(projection[mask].sum().item())
  counts[f"{name}/cmd_sum"] += float(cmd_mag[mask].sum().item())
  counts[f"{name}/actual_sum"] += float(actual_mag[mask].sum().item())


def run_evaluate_edge_cases(task_id: str, cfg: EvaluateEdgeConfig) -> dict[str, float]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=cfg.play)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.observations["policy"].enable_corruption = False

  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  arm_joint_ids, arm_joint_names = robot.find_joints(
    ["joint1", "joint2", "joint3"], preserve_order=True
  )
  print(f"[INFO] Using arm joints for edge-case eval: {arm_joint_names}")

  obs = env.get_observations()

  counts: dict[str, float] = {}
  for name in [
    "overall_active",
    "same_dir_right",
    "same_dir_left",
    "forward_full",
    "backward_full",
    "start_move_extreme",
    "stop_move_extreme",
  ]:
    counts[f"{name}/count"] = 0.0
    counts[f"{name}/stall_count"] = 0.0
    counts[f"{name}/undertrack_count"] = 0.0
    counts[f"{name}/projection_sum"] = 0.0
    counts[f"{name}/cmd_sum"] = 0.0
    counts[f"{name}/actual_sum"] = 0.0

  standing_cmd_count = 0.0
  total_samples = 0.0
  event_examples: list[str] = []
  prev_cmd_mag: torch.Tensor | None = None

  print(
    f"[INFO] Evaluating {cfg.num_steps} steps with {cfg.num_envs} envs "
    f"on {device} (play={cfg.play}, warmup_steps={cfg.warmup_steps})"
  )

  for step in range(cfg.num_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)

    if step < cfg.warmup_steps:
      continue

    command = env.unwrapped.command_manager.get_command("twist")
    assert command is not None

    actual_xy = robot.data.root_link_lin_vel_b[:, :2]
    cmd_xy = command[:, :2]
    cmd_mag = torch.norm(cmd_xy, dim=1)
    actual_mag = torch.norm(actual_xy, dim=1)

    arm_pos = robot.data.joint_pos[:, arm_joint_ids]
    base_yaw = arm_pos[:, 0]
    shoulder = arm_pos[:, 1]
    elbow = arm_pos[:, 2]

    active_cmd = cmd_mag > 0.25
    standing_cmd = cmd_mag < 0.05
    standing_cmd_count += float(standing_cmd.sum().item())
    total_samples += float(cmd_mag.numel())

    cmd_dir = cmd_xy / cmd_mag.clamp(min=1e-6).unsqueeze(1)
    projection = torch.sum(actual_xy * cmd_dir, dim=1)

    extended = elbow < -0.8
    same_dir_right = active_cmd & extended & (base_yaw > 1.5) & (cmd_xy[:, 1] > 0.25)
    same_dir_left = active_cmd & extended & (base_yaw < -1.5) & (cmd_xy[:, 1] < -0.25)
    forward_full = active_cmd & extended & (shoulder > 0.8) & (cmd_xy[:, 0] > 0.4)
    backward_full = active_cmd & extended & (shoulder < -0.8) & (cmd_xy[:, 0] < -0.2)
    extreme_pose = extended & ((torch.abs(base_yaw) > 1.5) | (torch.abs(shoulder) > 0.8))

    _case_stats(counts, "overall_active", active_cmd, cmd_mag, actual_mag, projection)
    _case_stats(counts, "same_dir_right", same_dir_right, cmd_mag, actual_mag, projection)
    _case_stats(counts, "same_dir_left", same_dir_left, cmd_mag, actual_mag, projection)
    _case_stats(counts, "forward_full", forward_full, cmd_mag, actual_mag, projection)
    _case_stats(counts, "backward_full", backward_full, cmd_mag, actual_mag, projection)

    if prev_cmd_mag is not None:
      start_move_extreme = extreme_pose & (prev_cmd_mag < 0.05) & (cmd_mag > 0.25)
      stop_move_extreme = extreme_pose & (prev_cmd_mag > 0.25) & (cmd_mag < 0.05)
      _case_stats(
        counts, "start_move_extreme", start_move_extreme, cmd_mag, actual_mag, projection
      )
      _case_stats(
        counts, "stop_move_extreme", stop_move_extreme, cmd_mag, actual_mag, projection
      )

      if len(event_examples) < 8:
        interesting = (active_cmd & (actual_mag < 0.10)).nonzero(as_tuple=False).flatten()
        for idx in interesting[: max(0, 8 - len(event_examples))]:
          i = int(idx.item())
          event_examples.append(
            "step="
            f"{step} env={i} cmd=({cmd_xy[i, 0]:+.2f},{cmd_xy[i, 1]:+.2f}) "
            f"actual=({actual_xy[i, 0]:+.2f},{actual_xy[i, 1]:+.2f}) "
            f"arm=(yaw={base_yaw[i]:+.2f}, shoulder={shoulder[i]:+.2f}, elbow={elbow[i]:+.2f})"
          )

    prev_cmd_mag = cmd_mag.clone()

    if (step + 1) % max(1, cfg.num_steps // 10) == 0:
      print(f"[INFO] Completed {step + 1}/{cfg.num_steps} steps")

  metrics: dict[str, float] = {
    "standing_command_fraction": standing_cmd_count / max(total_samples, 1.0),
  }

  for name in [
    "overall_active",
    "same_dir_right",
    "same_dir_left",
    "forward_full",
    "backward_full",
    "start_move_extreme",
    "stop_move_extreme",
  ]:
    count = counts[f"{name}/count"]
    metrics[f"{name}/count"] = count
    if count <= 0:
      metrics[f"{name}/stall_rate"] = float("nan")
      metrics[f"{name}/undertrack_rate"] = float("nan")
      metrics[f"{name}/mean_projection_over_cmd"] = float("nan")
      metrics[f"{name}/mean_actual_speed"] = float("nan")
      continue

    metrics[f"{name}/stall_rate"] = counts[f"{name}/stall_count"] / count
    metrics[f"{name}/undertrack_rate"] = counts[f"{name}/undertrack_count"] / count
    metrics[f"{name}/mean_projection_over_cmd"] = counts[f"{name}/projection_sum"] / max(
      counts[f"{name}/cmd_sum"], 1e-6
    )
    metrics[f"{name}/mean_actual_speed"] = counts[f"{name}/actual_sum"] / count

  print("\n" + "=" * 60)
  print("Velocity Edge-Case Evaluation")
  print("=" * 60)
  print(f"standing_command_fraction: {metrics['standing_command_fraction']:.4f}")
  for name in [
    "overall_active",
    "same_dir_right",
    "same_dir_left",
    "forward_full",
    "backward_full",
    "start_move_extreme",
    "stop_move_extreme",
  ]:
    print(f"\n{name}:")
    print(f"  count: {metrics[f'{name}/count']:.0f}")
    print(f"  stall_rate: {metrics[f'{name}/stall_rate']:.4f}")
    print(f"  undertrack_rate: {metrics[f'{name}/undertrack_rate']:.4f}")
    print(f"  mean_projection_over_cmd: {metrics[f'{name}/mean_projection_over_cmd']:.4f}")
    print(f"  mean_actual_speed: {metrics[f'{name}/mean_actual_speed']:.4f}")

  if event_examples:
    print("\nExample stall-like events:")
    for example in event_examples:
      print(f"  {example}")
  print("=" * 60)

  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
      json.dump(
        {
          "metrics": metrics,
          "examples": event_examples,
        },
        f,
        indent=2,
      )
    print(f"[INFO] Saved metrics to {output_path}")

  env.close()
  return metrics


def main() -> None:
  import mjlab.tasks  # noqa: F401

  velocity_tasks = [t for t in list_tasks() if "Velocity" in t]
  if not velocity_tasks:
    print("No velocity tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    EvaluateEdgeConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  run_evaluate_edge_cases(chosen_task, args)


if __name__ == "__main__":
  main()
