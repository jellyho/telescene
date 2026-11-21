# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""A G1 (TriHand) at a table, driven by SONIC, for headset teleop. Isaac Lab 3.0, one environment.

Action, 122-D: the 14 TriHand finger targets, then SONIC in ``switch`` mode (108-D, see
:class:`telescene.sonic.action_cfg.SonicActionCfg`). :class:`telescene.sonic.controllers.SonicSwitchController` fills it.

The scene is Isaac Lab's own (ground, dome light, the packing table from the Isaac Nucleus) plus a 1 kg box on the table.
Isaac downloads the table on first use.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg

from telescene.sonic.action_cfg import SonicActionCfg
from telescene.sonic.actuators import SONIC_INIT_JOINT_POS, SONIC_INIT_ROOT_HEIGHT, with_sonic_actuators
from telescene.sonic.core.checkpoint import find_sonic_dir

TASK_ID = "Telescene-G1-Sonic-v0"

TRIHAND_JOINT_NAMES = (
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
)
"""The order of ``telescene.sonic.inputs.HAND_OUTPUT_ORDER``: column i drives joint i."""


def _robot() -> ArticulationCfg:
    robot = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot = replace(robot, init_state=replace(robot.init_state, pos=(0.0, 0.0, SONIC_INIT_ROOT_HEIGHT),
                                              joint_pos=dict(SONIC_INIT_JOINT_POS)))
    # SONIC was trained with self-collision on; Isaac Lab's G1 ships with it off.
    for props in robot.spawn.articulation_props:
        props.enabled_self_collisions = True
    return with_sonic_actuators(robot, keep=("hands",))


@configclass
class G1SonicSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _robot()
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, -0.3)),  # where Isaac Lab's G1 pick-place task has it
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
                         rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)))
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.45, 0.72)),  # on the table top (z 0.695)
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.05, 0.05), rigid_props=RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), collision_props=CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.40, 0.08))))


@configclass
class ActionsCfg:
    hands = JointPositionActionCfg(asset_name="robot", joint_names=list(TRIHAND_JOINT_NAMES), preserve_order=True,
                                   scale=1.0, use_default_offset=False)
    """Absolute finger targets, in TRIHAND_JOINT_NAMES order (``preserve_order``)."""

    sonic = SonicActionCfg(asset_name="robot", mode="switch", vr_calibrate_wrists=True, background_planner=True)
    """``onnx_dir`` is filled in ``__post_init__``. A replan takes about 70 ms on CPU, so it runs in the background."""


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    robot_fallen = DoneTerm(func=mdp.root_height_below_minimum,
                            params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot")})


@configclass
class G1SonicEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1SonicSceneCfg = G1SonicSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    sonic_onnx_dir: str | None = None
    """The three SONIC ONNX files. None: ``$TELESCENE_SONIC_DIR``, else ``checkpoints/gear_sonic/``."""

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1 / 200            # 200 Hz physics, 50 Hz control: SONIC's period
        self.sim.render_interval = 4     # nothing is rendered on the PC; the headset draws the scene
        self.episode_length_s = 3600.0
        # Isaac's default friction is 0.5; SONIC's training (mjlab) uses 1.0, and the hands grip better with it.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.actions.sonic.onnx_dir = str(find_sonic_dir(self.sonic_onnx_dir))


def register() -> None:
    import gymnasium as gym

    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:G1SonicEnvCfg"})
