"""K1 soccer-kick tasks (k1_kick_soccer_rand_v1 'fixed' trial exports).

Each trial ONNX has one goal layout baked in; see the policy's DEPLOY.md for
how to place the feet, ball and goal.  Gains, default pose and action scale
below are copied from the ONNX metadata and checked against it at load time.
"""
from booster_deploy.controllers.controller_cfg import ControllerCfg
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...kick_soccer import KickPolicyCfg


@configclass
class K1KickSoccerControllerCfg(ControllerCfg):
    # Joint order: head(2), left arm(4), right arm(4), left leg(6), right leg(6).
    robot = K1_CFG.replace(  # type: ignore
        # Training spawn pose (trunk ~0.495 m): knees bent, arms out.
        default_joint_pos=[
            0.0, 0.7,
            0.0, -1.4, 0.0, -0.4,
            0.0, 1.4, 0.0, 0.4,
            -0.525, 0.0, 0.0, 1.05, -0.525, 0.0,
            -0.525, 0.0, 0.0, 1.05, -0.525, 0.0,
        ],
        joint_stiffness=[
            4.0, 4.0,
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
            80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
            80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
        ],
        # NOTE: these are the training-simulator Kd values.  The README warns
        # that parallel-actuated joints (ankles) take motor-side Kd
        # (Kd = 2*zeta*J_eq*2*pi*f_n); retune here and set
        # policy.allow_gain_override=True if hardware needs it.
        joint_damping=[
            0.25, 0.25,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        ],
    )
    vel_command = None
    policy: KickPolicyCfg = KickPolicyCfg(
        action_scale=[
            0.375, 0.375,
            0.35, 0.35, 0.35, 0.35,
            0.35, 0.35, 0.35, 0.35,
            0.213, 0.237, 0.120, 0.350, 0.191, 0.191,
            0.213, 0.237, 0.120, 0.350, 0.191, 0.191,
        ],
    )

    def __post_init__(self):
        super().__post_init__()
        # Spawn MuJoCo at the training trunk height for the crouched pose.
        self.mujoco.init_pos = [0.0, 0.0, 0.5]


@configclass
class K1KickSoccerD150B00MiddleCfg(K1KickSoccerControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/k1/models/d1.50_b+00_middle.onnx"


@configclass
class K1KickSoccerD150B00LeftCfg(K1KickSoccerControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/k1/models/d1.50_b+00_left.onnx"


@configclass
class K1KickSoccerD150B00RightCfg(K1KickSoccerControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/k1/models/d1.50_b+00_right.onnx"


register_task("k1_kick_soccer_d150_b00_middle", K1KickSoccerD150B00MiddleCfg())
register_task("k1_kick_soccer_d150_b00_left", K1KickSoccerD150B00LeftCfg())
register_task("k1_kick_soccer_d150_b00_right", K1KickSoccerD150B00RightCfg())
