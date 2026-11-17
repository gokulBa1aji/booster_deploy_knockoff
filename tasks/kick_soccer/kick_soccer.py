"""Single-shot soccer kick policy (k1_kick_soccer_rand_v1 trial exports).

The trial ONNX files have the goal layout baked in and take the standard
72-value proprioceptive observation:

    [base_ang_vel(3), projected_gravity(3), joint_pos - default(22),
     joint_vel(22), last_action(22)]

with no scaling, clipping or history (the input normalizer is inside the
graph).  Output is 22 raw actions; targets are
``default_joint_pos + action_scale * action``.

Runtime phases (counted in policy steps at ``policy_dt``):

1. ramp:  interpolate from the pose at task start to ``default_joint_pos``
          (the training spawn pose, a low crouch) over ``ramp_s``.
2. kick:  run the ONNX for ``episode_s`` (the trained episode length).
3. hold:  ramp to ``hold_pose`` over ``ramp_s`` and hold it until the
          operator stops the task.

The trained (soft) PD gains cannot hold the crouch on their own (the robot
tips forward in ~1 s in sim), so by default the ramp and hold phases use the
robot's stiffer ``prepare_state`` gains and the trained gains are switched
in only while the ONNX runs.
"""

from __future__ import annotations

from dataclasses import MISSING
import os

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


class KickPolicy(Policy):
    """Ramp to the start pose, run the kick ONNX once, then hold."""

    def __init__(self, cfg: KickPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.device = torch.device(cfg.device)

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, policy_path)
        if os.path.splitext(policy_path)[1].lower() != ".onnx":
            raise ValueError("KickPolicy expects an .onnx trial export")
        self._model = create_policy_runner(policy_path, self.device)

        self.num_joints = self.robot.num_joints
        self.action_scale = torch.as_tensor(
            cfg.action_scale, dtype=torch.float32, device=self.device)
        if self.action_scale.numel() != self.num_joints:
            raise ValueError(
                f"action_scale has {self.action_scale.numel()} entries; "
                f"robot has {self.num_joints} joints")

        self.robot.data.to(self.device)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)
        self.last_action = torch.zeros(
            self.num_joints, dtype=torch.float32, device=self.device)
        self.ramp_start_pos = self.default_joint_pos.clone()
        self.last_target = self.default_joint_pos.clone()
        if cfg.hold_pose == "prepare":
            # Straight-legged stance: statically stable, unlike the crouch.
            self.hold_pos = torch.tensor(
                self.robot.cfg.prepare_state.joint_pos,
                dtype=torch.float32, device=self.device)
        elif cfg.hold_pose == "default":
            self.hold_pos = self.default_joint_pos.clone()
        else:
            raise ValueError(
                f"hold_pose must be 'prepare' or 'default', got "
                f"{cfg.hold_pose!r}")
        self.gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        self.ramp_steps = max(1, round(cfg.ramp_s / controller.cfg.policy_dt))
        self.kick_steps = round(cfg.episode_s / controller.cfg.policy_dt)
        self._step = 0
        self._phase = None

        # Gains: trained ones for the kick, optionally stiffer ones for
        # ramp/hold.  The controllers read robot.joint_stiffness/damping on
        # every ctrl_step, so updating them in place switches the gains.
        self.policy_kp = self.robot.joint_stiffness.clone()
        self.policy_kd = self.robot.joint_damping.clone()
        if cfg.hold_gains == "prepare":
            ps = self.robot.cfg.prepare_state
            self.hold_kp = torch.tensor(ps.stiffness, dtype=torch.float32)
            self.hold_kd = torch.tensor(ps.damping, dtype=torch.float32)
        elif cfg.hold_gains == "policy":
            self.hold_kp, self.hold_kd = self.policy_kp, self.policy_kd
        else:
            raise ValueError(
                f"hold_gains must be 'prepare' or 'policy', got "
                f"{cfg.hold_gains!r}")

        self._validate_against_metadata()

    # ------------------------------------------------------------------
    # Setup checks
    # ------------------------------------------------------------------
    def _validate_against_metadata(self) -> None:
        """Refuse to run if the robot cfg disagrees with the ONNX metadata."""
        session = self._model.session
        inputs = session.get_inputs()
        expected_obs = 6 + 3 * self.num_joints
        if inputs[0].shape[-1] != expected_obs:
            raise ValueError(
                f"ONNX input has {inputs[0].shape[-1]} values; expected "
                f"{expected_obs}. Only the 'fixed' trial exports are supported.")

        meta = session.get_modelmeta().custom_metadata_map
        if meta.get("kick_variant", "fixed") != "fixed":
            raise ValueError(
                f"kick_variant={meta.get('kick_variant')!r}; only 'fixed' "
                "trial exports (single obs input) are supported")

        joint_names = meta.get("joint_names", "").split(",")
        if len(joint_names) != self.num_joints:
            raise ValueError(
                f"ONNX lists {len(joint_names)} joints; robot has "
                f"{self.num_joints}")
        # Metadata uses Booster's display names (Head_Yaw, Left_Hip_Pitch, ...)
        # while the robot cfg uses URDF names; check they line up in order.
        for meta_name, cfg_name in zip(joint_names, self.robot.cfg.joint_names):
            key = cfg_name.lower().lstrip("a").replace("_joint", "")
            if meta_name.lower() != key:
                raise ValueError(
                    f"joint order mismatch: ONNX {meta_name!r} vs robot "
                    f"{cfg_name!r}")

        checks = {
            "default_joint_pos": self.robot.cfg.default_joint_pos,
            "joint_stiffness": self.robot.cfg.joint_stiffness,
            "joint_damping": self.robot.cfg.joint_damping,
            "action_scale": list(self.cfg.action_scale),
        }
        if self.cfg.allow_gain_override:
            print("[KickPolicy] WARNING: kp/kd differ from training "
                  "(allow_gain_override=True)")
            checks.pop("joint_stiffness")
            checks.pop("joint_damping")
        for key, cfg_vals in checks.items():
            if key not in meta:
                raise ValueError(f"ONNX metadata is missing {key!r}")
            meta_vals = torch.tensor(_parse_floats(meta[key]))
            cfg_t = torch.tensor(cfg_vals, dtype=torch.float32)
            if meta_vals.shape != cfg_t.shape or not torch.allclose(
                    meta_vals, cfg_t, atol=1e-3):
                raise ValueError(
                    f"{key} in the task cfg does not match the ONNX metadata\n"
                    f"  cfg:  {cfg_t.tolist()}\n  onnx: {meta_vals.tolist()}")

        layout = meta.get("kick_layout")
        if layout:
            print(f"[KickPolicy] layout: {layout}")

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._step = 0
        self._phase = None
        self.last_action.zero_()
        self.ramp_start_pos = self.robot.data.joint_pos.to(self.device).clone()
        self.last_target.copy_(self.default_joint_pos)

    def _set_phase(self, phase: str, message: str) -> None:
        if self._phase != phase:
            self._phase = phase
            kp, kd = ((self.policy_kp, self.policy_kd) if phase == "kick"
                      else (self.hold_kp, self.hold_kd))
            self.robot.joint_stiffness.copy_(kp)
            self.robot.joint_damping.copy_(kd)
            print(f"\n[KickPolicy] {message}")

    def _projected_gravity(self) -> torch.Tensor:
        return lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w, self.gravity_w)

    def _safety_check(self, projected_gravity: torch.Tensor) -> None:
        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print(
                "\nFalling detected, stopping policy for safety. "
                "You can disable safety fallback by setting "
                f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                "to False."
            )
            self.controller.stop()

    def compute_observation(self, projected_gravity: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.robot.data.root_ang_vel_b,
            projected_gravity,
            self.robot.data.joint_pos - self.default_joint_pos,
            self.robot.data.joint_vel,
            self.last_action,
        ], dim=0)

    def inference(self) -> torch.Tensor:
        step = self._step
        self._step += 1
        projected_gravity = self._projected_gravity()
        self._safety_check(projected_gravity)

        # Phase 1: ramp to the training start pose.
        if step < self.ramp_steps:
            self._set_phase(
                "ramp", f"Moving to kick start pose over {self.cfg.ramp_s:.2f} s")
            alpha = (step + 1) / self.ramp_steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            return torch.lerp(self.ramp_start_pos, self.default_joint_pos, alpha)

        # Phase 2: run the kick policy for the trained episode length.
        kick_step = step - self.ramp_steps
        if kick_step < self.kick_steps:
            self._set_phase(
                "kick", f"Kick policy running for {self.cfg.episode_s:.1f} s")
            obs = self.compute_observation(projected_gravity)
            if kick_step == 0:
                torch.set_printoptions(precision=3, sci_mode=False)
                print(
                    "[KickPolicy] first obs (expect gravity ~ (-0.02, 0, -1), "
                    "joint terms ~ 0):\n"
                    f"  ang_vel   {obs[0:3].tolist()}\n"
                    f"  gravity   {obs[3:6].tolist()}\n"
                    f"  joint_pos {obs[6:6 + self.num_joints].tolist()}")
            with torch.no_grad():
                action = self._model(obs).reshape(-1)
            if action.numel() != self.num_joints:
                raise RuntimeError(
                    f"policy returned {action.numel()} actions; expected "
                    f"{self.num_joints}")
            self.last_action.copy_(action)
            target = self.default_joint_pos + self.action_scale * action
            self.last_target.copy_(target)
            return target

        # Phase 3: ramp to the hold pose, then hold it until the operator
        # stops the task.
        self._set_phase(
            "hold",
            f"Kick episode finished; moving to {self.cfg.hold_pose!r} pose and "
            "holding. Stop the task (remote / Ctrl-C) when ready.")
        hold_step = kick_step - self.kick_steps
        alpha = min(1.0, (hold_step + 1) / self.ramp_steps)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return torch.lerp(self.last_target, self.hold_pos, alpha)


@configclass
class KickPolicyCfg(PolicyCfg):
    constructor = KickPolicy
    checkpoint_path: str = MISSING  # type: ignore
    # Per-joint action scale in robot joint order (from ONNX metadata).
    action_scale: list[float] = MISSING  # type: ignore
    # Time to interpolate from the hand-off pose to default_joint_pos.  The
    # policy should start within ~1 s of reaching that pose.
    ramp_s: float = 1.0
    # Trained episode length (metadata ``kick_episode_s``).
    episode_s: float = 10.0
    # Set True to deliberately run with kp/kd that differ from training
    # (e.g. retuned motor-side Kd for the parallel ankles).
    allow_gain_override: bool = False
    # PD gains for the ramp and hold phases: "prepare" uses the robot's stiff
    # prepare_state gains, "policy" keeps the trained gains throughout.
    hold_gains: str = "prepare"
    # Pose held after the episode: "prepare" (robot prepare_state pose, a
    # straight-legged stance) or "default" (the crouch; tips over in sim).
    hold_pose: str = "prepare"
