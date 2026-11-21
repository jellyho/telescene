# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""A G1 (TriHand) at a table, for headset teleop. Isaac Lab 3.0, one environment. Two tasks:

- ``Telescene-G1-Sonic-v0``: driven by SONIC. Action, 122-D: the 14 TriHand finger targets, then SONIC in ``switch``
  mode (108-D, see :class:`telescene.sonic.action_cfg.SonicActionCfg`), which
  :class:`telescene.sonic.controllers.SonicSwitchController` fills.
- ``Telescene-G1-Tracker-v0``: driven by a whole-body tracker (``telescene.trackers``). Action, 43-D: the 14 finger
  targets, then 29 absolute joint targets in ``telescene.trackers.supermimic.JOINT_NAMES`` order. The body has the
  motors SuperMimic was trained with (mjlab's); ``use_tracker_motors`` gives it those of another tracker's spec.

The scene is Isaac Lab's own (ground, dome light, the packing table from the Isaac Nucleus) plus a 1 kg box on the table.
Isaac downloads the table on first use. ``busybox.use_busybox`` swaps it for a BusyBox scene.
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
TRACKER_TASK_ID = "Telescene-G1-Tracker-v0"

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
    # 0.3 m back from the table, whose front edge is at y 0.15: at y 0 the hands start inside it.
    robot = replace(robot, init_state=replace(robot.init_state, pos=(0.0, -0.3, SONIC_INIT_ROOT_HEIGHT),
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


# ------------------------------------------------------------------------------------------- tracker task
def tracker_robot(robot: ArticulationCfg) -> ArticulationCfg:
    """``robot`` with mjlab's motors (SuperMimic's training) on the 29 body joints and its default pose. No velocity
    limit, as in mjlab. The hands keep theirs."""
    from isaaclab.actuators import ImplicitActuatorCfg

    from telescene.trackers.supermimic import ARMATURE, DAMPING, DEFAULT_JOINT_POS, EFFORT_LIMIT, JOINT_NAMES, STIFFNESS

    def per_joint(values):
        return {n: float(v) for n, v in zip(JOINT_NAMES, values)}

    body = ImplicitActuatorCfg(joint_names_expr=list(JOINT_NAMES), stiffness=per_joint(STIFFNESS),
                               damping=per_joint(DAMPING), armature=per_joint(ARMATURE),
                               effort_limit_sim=per_joint(EFFORT_LIMIT), velocity_limit_sim=1000.0)
    init = replace(robot.init_state, pos=(robot.init_state.pos[0], robot.init_state.pos[1], 0.76),  # mjlab's knees bent
                   joint_pos=per_joint(DEFAULT_JOINT_POS))
    return replace(robot, actuators={"hands": robot.actuators["hands"], "body": body}, init_state=init)


def tracker_runtime(spec: str) -> dict:
    """A tracker spec's ``runtime`` block (``file.json#variant`` for one of several), with its ``slug``."""
    import json
    import os

    path, _, variant = spec.partition("#")
    with open(path) as f:
        raw = json.load(f)
    raw = raw.get("runtime", raw)
    if variant:
        raw = raw[variant]
    return {"slug": os.path.splitext(os.path.basename(path))[0] + (f"_{variant}" if variant else ""), **raw}


MOTOR_FIELDS = ("kp", "kd", "armature", "effort", "velocity_limit")


def use_tracker_motors(env_cfg, spec: str) -> None:
    """Give the body the motors, default pose and timing a tracker's spec says it was trained with. What the spec leaves
    null stays mjlab's, and this says so: a tracker driven through motors it was not trained on says little about it."""
    import numpy as np
    from isaaclab.actuators import ImplicitActuatorCfg

    from telescene.trackers.onnx_tracker import Kinematics
    from telescene.trackers.supermimic import ARMATURE, DAMPING, EFFORT_LIMIT, JOINT_NAMES, STIFFNESS

    runtime = tracker_runtime(spec)
    names = list(runtime["joint_names"])
    mj = {n: i for i, n in enumerate(JOINT_NAMES)}
    fallback = {"kp": STIFFNESS, "kd": DAMPING, "armature": ARMATURE, "effort": EFFORT_LIMIT, "velocity_limit": None}
    values, kept = {}, []
    for name in MOTOR_FIELDS:
        given = runtime.get(name)
        if given is None:
            kept.append(name)
            given = None if fallback[name] is None else [float(fallback[name][mj[n]]) for n in names]
        elif not isinstance(given, list):
            given = [float(given)] * len(names)
        values[name] = None if given is None else {n: float(v) for n, v in zip(names, given)}
    if kept:
        print(f"[tracker] {runtime['slug']}: the spec gives no {', '.join(kept)}; using mjlab's", flush=True)
    robot = env_cfg.scene.robot
    body = ImplicitActuatorCfg(joint_names_expr=names, stiffness=values["kp"], damping=values["kd"],
                               armature=values["armature"], effort_limit_sim=values["effort"],
                               velocity_limit_sim=values["velocity_limit"] or 1000.0)
    z = runtime.get("spawn_root_height")
    if z is None:  # standing on the default pose's own feet
        try:
            q = dict(zip(names, runtime["default_joint_pos"]))
            z = Kinematics().stand_height(np.array([q[n] for n in JOINT_NAMES])) + 0.003
        except (ImportError, OSError, ValueError):
            z = robot.init_state.pos[2]
    init = replace(robot.init_state, pos=(robot.init_state.pos[0], robot.init_state.pos[1], float(z)),
                   joint_pos={n: float(v) for n, v in zip(names, runtime["default_joint_pos"])})
    env_cfg.scene.robot = replace(robot, actuators={"hands": robot.actuators["hands"], "body": body}, init_state=init)
    control_dt = float(runtime.get("control_dt") or 0.02)
    env_cfg.sim.dt = float(runtime.get("sim_dt") or 0.005)
    env_cfg.decimation = int(runtime.get("decimation") or round(control_dt / env_cfg.sim.dt))
    env_cfg.sim.render_interval = env_cfg.decimation
    if abs(env_cfg.sim.dt * env_cfg.decimation - control_dt) > 1e-6:
        raise ValueError(f"{runtime['slug']}: sim_dt {env_cfg.sim.dt} x decimation {env_cfg.decimation} is not {control_dt}")
    if runtime.get("self_collision") is not None:
        for props in env_cfg.scene.robot.spawn.articulation_props:
            props.enabled_self_collisions = bool(runtime["self_collision"])


@configclass
class TrackerActionsCfg:
    hands = JointPositionActionCfg(asset_name="robot", joint_names=list(TRIHAND_JOINT_NAMES), preserve_order=True,
                                   scale=1.0, use_default_offset=False)
    body = JointPositionActionCfg(asset_name="robot", joint_names=[], preserve_order=True, scale=1.0,
                                  use_default_offset=False)
    """Absolute targets in ``telescene.trackers.supermimic.JOINT_NAMES`` order, filled in ``__post_init__``."""


@configclass
class G1TrackerEnvCfg(G1SonicEnvCfg):
    actions: TrackerActionsCfg = TrackerActionsCfg()

    def __post_init__(self):
        from telescene.trackers.supermimic import JOINT_NAMES

        self.decimation = 4
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4
        self.episode_length_s = 3600.0
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.scene.robot = tracker_robot(self.scene.robot)
        self.actions.body.joint_names = list(JOINT_NAMES)


# ------------------------------------------------------------------------------------------- grip
GRIP_COLLIDERS = r"/Robot/.*_hand_|/BusyBox/Geometry/obj_"
"""Colliders that get a torsional patch: the hands' and the BusyBox scenes' free objects'."""


def torsion_patch(env, env_ids, radius: float, paths: str = GRIP_COLLIDERS) -> None:
    """Prestartup event: a PhysX torsional patch of ``radius`` [m] on every collider whose path matches ``paths``.

    A pinch is two point contacts, and PhysX gives a point contact no friction about its normal, so a small object
    turns freely between the fingers. The patch adds that friction, as MuJoCo's torsional friction does. The G1's
    colliders are instanced, and an instance cannot be edited, so the matching ones are made local first.
    """
    import re

    from pxr import PhysxSchema, Usd, UsdPhysics

    stage, match = env.sim.stage, re.compile(paths)
    for _ in range(4):  # instances can nest
        found = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
                 if p.IsInstance() and match.search(p.GetPath().pathString)]
        if not found:
            break
        for prim in found:
            stage.GetPrimAtPath(prim.GetPath()).SetInstanceable(False)
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and match.search(prim.GetPath().pathString):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateTorsionalPatchRadiusAttr(radius)
            api.CreateMinTorsionalPatchRadiusAttr(radius / 2)
            count += 1
    print(f"[telescene] torsional patch {radius * 1000:.0f} mm on {count} colliders (hands and objects)", flush=True)


def add_torsion_patch(env_cfg, radius: float = 0.01) -> None:
    """Run ``torsion_patch`` before the simulator starts. 0 leaves it off."""
    if radius > 0:
        env_cfg.scene.replicate_physics = False  # a prestartup USD edit needs it off; teleop runs one env anyway
        env_cfg.events.grip_torsion_patch = EventTerm(func=torsion_patch, mode="prestartup", params={"radius": radius})


def register() -> None:
    import gymnasium as gym

    for task, cfg in ((TASK_ID, "G1SonicEnvCfg"), (TRACKER_TASK_ID, "G1TrackerEnvCfg")):
        if task not in gym.registry:
            gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                         kwargs={"env_cfg_entry_point": f"{__name__}:{cfg}"})
