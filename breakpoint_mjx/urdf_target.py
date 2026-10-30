"""Bring-your-own-robot: turn a ROS/ROS2 **URDF description** into a BreakPoint ``Target``.

Every other target in this repo names a MuJoCo Playground env id, which means BreakPoint
can only test the handful of robots Playground ships. Most teams do not have a Playground
env -- they have a URDF (the ROS robot description their whole stack already uses). This
module removes that adoption barrier: point ``make_urdf_target`` at a ``.urdf`` and you get
a real ``Target`` the existing search / evaluator / atlas / render pipeline consumes
unchanged.

WHY THIS IS NOT A ONE-LINER
---------------------------
``mujoco.MjModel.from_xml_path("robot.urdf")`` does load a URDF, but the resulting model is
NOT simulatable as a robot. Three things are missing, all measured on mujoco 3.10:

1. **No actuators.** URDF ``<transmission>`` / ``ros2_control`` blocks are not imported, so
   ``nu == 0``. A policy has nothing to drive.
2. **No floating base, and the root link's mass is DISCARDED.** A URDF root link has no
   joint to its parent, so MuJoCo *fuses the root link into the worldbody*: the robot is
   welded to the world AND its trunk mass vanishes from ``body_mass.sum()`` (measured: an
   8.2 kg fixture imported as 3.2 kg). Adding a ``<freejoint/>`` to the emitted MJCF does
   not fix this -- by then the root body no longer exists.
3. **No world.** No ground plane, no light, no tracking camera, no solver/timestep choice.

THE PIPELINE (five stages, each fail-loud)
------------------------------------------
``urdf_path``
  -> (1) **parse + audit the URDF** (ElementTree): find the single root link, classify every
     joint, reject constructs MuJoCo silently drops (``<mimic>``) or silently expands
     (``planar``), resolve ``package://`` mesh URIs to absolute paths (MuJoCo does NOT
     resolve them -- measured), and check total mass / xacro leftovers.
  -> (2) **pre-inject a URDF-level floating joint**: a massless ``bp_world`` link plus
     ``<joint type="floating">`` to the root link. This is the fix for (2) above: with a
     joint to its parent the root link stays a real body, keeps its inertial, and compiles
     to ``jnt_type[0] == mjJNT_FREE`` -- exactly the layout ``validate_legged_model``
     requires. Skipped when ``floating_base=False`` (a bolted-down arm).
  -> (3) **compile once and emit MJCF** (``MjSpec.to_xml()``, falling back to
     ``mj_saveLastXML``).
  -> (4) **inject the missing world into the MJCF** (ElementTree, not string surgery):
     ``<option>`` solver/timestep, a named ``floor`` plane geom with friction, a light, a
     ``track`` camera on the root body, and one ``<position>`` actuator per actuated joint
     whose ``ctrlrange`` is the URDF joint limit and whose ``forcerange`` is the URDF
     effort limit. (Optionally also ``armature``/``frictionloss`` -- URDF cannot express
     rotor inertia, so two theta dims are dead no-ops without them; the report says so.)
  -> (5) **recompile, validate, and MEASURE**: settle the robot under gravity holding its
     default pose and read the real standing height, which is what the derived theta-space
     bounds and the ``trunk_height_min`` fall threshold are scaled from -- never copied
     from Go1.

PUBLIC API
----------
* ``urdf_to_mjcf(urdf_path, **opts) -> (mjcf_text, info)`` -- stages 1-4, pure XML. No jax.
* ``load_urdf_model(urdf_path, **opts) -> (mj_model, report)`` -- stages 1-5. No jax.
* ``URDFEnv(mj_model, report, ...)`` -- the minimal env implementing the ``rollout_one``
  contract (``reset`` / ``step`` / ``obs`` / ``mjx_model`` / ``observation_size`` /
  ``action_size``, plus ``dt`` and ``render`` for the renderer). Needs jax + mjx.
* ``derive_theta_space(report, ...) -> ThetaSpace`` and ``write_theta_space_yaml`` --
  bounds scaled from the measured robot, not from Go1.
* ``make_urdf_target(name, urdf_path, ...) -> Target`` -- the product entry point.

HONESTY: THE OBSERVATION IS *BREAKPOINT'S*, NOT YOUR POLICY'S
-------------------------------------------------------------
``URDFEnv`` synthesizes an observation from the model (see ``URDFEnv`` for the exact
layout). A URDF says nothing about what a policy was trained to see, so this layout is a
BreakPoint *convention* -- it is almost certainly NOT the vector your trained policy
expects. Feeding your ONNX/torch export straight in will produce garbage actions that look
like a working simulation. Map your policy's ordering explicitly with an
``ObsSpec`` (``breakpoint_mjx.adapters.obs_spec``); ``env.obs_layout`` exposes the
``(name, slice)`` blocks programmatically so you can build that mapping. The report repeats
this warning under ``report["obs_convention_warning"]``.

Import-safety: this module imports with NO jax and NO mujoco installed. ``mujoco`` is
imported inside the loader functions, ``jax``/``mjx`` inside ``URDFEnv``. Only ``numpy`` /
``xml.etree`` / the repo's own import-safe config+target modules are imported at top level.
"""
from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from breakpoint_mjx.config import ThetaDim, ThetaSpace
from breakpoint_mjx.paths import config_path

# ---------------------------------------------------------------------------- #
# Constants / defaults
# ---------------------------------------------------------------------------- #

#: URDF joint types that become a single actuated 1-dof MuJoCo joint.
ACTUATED_URDF_JOINT_TYPES = ("revolute", "continuous", "prismatic")
#: URDF joint types that are structural (no dof) and need no actuator.
PASSIVE_URDF_JOINT_TYPES = ("fixed",)
#: URDF joint types this importer REFUSES. ``planar`` is refused because MuJoCo silently
#: expands it into three separate joints (measured: ``j1`` -> ``j1_TX``/``j1_TY``/``j1_RZ``),
#: which breaks the "one dof per actuated joint" layout the legged field math slices on.
REFUSED_URDF_JOINT_TYPES = ("planar",)

#: Name given to the injected ground plane. Matches ``DEFAULT_FLOOR_CANDIDATES``.
FLOOR_GEOM_NAME = "floor"
#: Name given to the injected tracking camera. Matches ``Target.camera``'s default.
TRACK_CAMERA_NAME = "track"
#: Name of the massless helper link used to give the root link a parent (stage 2).
_WORLD_LINK_NAME = "bp_urdf_world"

#: Default sliding/torsional/rolling friction for the injected floor. The first entry is
#: the nominal the ``friction_floor`` theta dim perturbs; 0.8 matches the Go1 scene so the
#: derived space lines up with ``configs/legged_default/theta_space.yaml`` at nominal.
DEFAULT_FLOOR_FRICTION = (0.8, 0.005, 0.0001)

#: Position-error (rad for hinges, m for slides) the default per-joint gain is sized to
#: hold at the joint's full effort limit: ``kp = effort / error``. A 24 N.m joint therefore
#: gets kp ~= 69, the same order as the Go1 MJCF's kp=35 on a 23.7 N.m joint.
DEFAULT_KP_ERROR_RAD = 0.35
DEFAULT_KP_ERROR_M = 0.05
#: Damping as a fraction of kp when ``kd`` is not given.
DEFAULT_KD_RATIO = 0.05
#: Clamp on derived gains, so a URDF with a nonsense effort limit cannot produce a model
#: that explodes on step 1 (or one that cannot hold its own weight).
KP_BOUNDS = (1.0, 2000.0)
#: Gain used when a joint declares no usable effort limit.
FALLBACK_KP = 60.0

#: MJX-friendly solver settings injected into ``<option>``. Newton + few iterations is the
#: MJX locomotion convention; every value is overridable via ``load_urdf_model(option=...)``.
DEFAULT_OPTION = {
    "timestep": "0.002",
    "solver": "Newton",
    "iterations": "4",
    "ls_iterations": "8",
    "integrator": "implicitfast",
    "cone": "pyramidal",
}

#: The loud, repeated warning about the synthesized observation (see module docstring).
OBS_CONVENTION_WARNING = (
    "The observation vector is BREAKPOINT'S convention derived from the URDF, NOT your "
    "trained policy's. A URDF does not record what a policy was trained to see, so this "
    "ordering/scaling is a guess that is almost certainly wrong for your export. Map your "
    "policy's expected ordering with breakpoint_mjx.adapters.obs_spec.ObsSpec using "
    "env.obs_layout; do NOT assume an arbitrary ONNX policy will just work."
)


class URDFImportError(ValueError):
    """Raised when a URDF cannot be turned into a physically meaningful MuJoCo model.

    Always carries the specific offending names/paths, because the whole point of this
    importer is that a silently-wrong model (no contacts, no mass, an ignored mimic joint)
    is worse than a hard failure.
    """


# ---------------------------------------------------------------------------- #
# Stage 1 -- parse + audit the URDF
# ---------------------------------------------------------------------------- #

def _float(node, attr, default=None):
    if node is None or node.get(attr) is None:
        return default
    return float(node.get(attr))


def _resolve_mesh(filename, urdf_dir, meshdir):
    """Resolve one URDF mesh ``filename`` to an absolute existing path, or ``None``.

    Handles the three forms found in the wild:
      * ``package://<pkg>/<rel>``  -- ROS package URI. MuJoCo does NOT resolve these
        (measured: "Error opening file 'package://...'"), so we do it here by searching
        ``meshdir`` and the URDF's own directory (and its parent, the usual package root)
        for ``<pkg>/<rel>`` and for ``<rel>``.
      * ``file:///abs/path``       -- stripped to the path.
      * a plain relative/absolute path -- resolved against ``meshdir`` then ``urdf_dir``.
    """
    raw = filename.strip()
    if raw.startswith("file://"):
        raw = raw[len("file://"):]

    roots = []
    if meshdir:
        roots.append(Path(meshdir))
    roots += [urdf_dir, urdf_dir.parent]

    if raw.startswith("package://"):
        rest = raw[len("package://"):]
        pkg, _, rel = rest.partition("/")
        candidates = []
        for r in roots:
            candidates += [r / pkg / rel, r / rel, r / rest]
    else:
        p = Path(raw)
        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [r / raw for r in roots]

    for c in candidates:
        if c.is_file():
            return str(c.resolve()).replace("\\", "/")
    return None


def _audit_urdf(urdf_path, *, meshdir=None):
    """Parse the URDF and return an audit dict; raise ``URDFImportError`` on anything
    that would compile into a physically meaningless model.

    Returns a dict with: ``tree`` (mutable ElementTree), ``robot_name``, ``root_link``,
    ``links``, ``joints`` (list of dicts), ``urdf_total_mass_kg``, ``meshes``,
    ``unsupported`` and ``warnings``.
    """
    path = Path(urdf_path)
    if not path.is_file():
        raise URDFImportError(f"URDF not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    # xacro is a TEMPLATE language; mujoco cannot expand it and neither can we.
    if path.suffix.lower() == ".xacro" or "xacro" in text[:4000] or "${" in text:
        raise URDFImportError(
            f"{path.name} looks like a xacro TEMPLATE (xacro namespace or ${{...}} "
            f"substitutions present), not a flat URDF. Expand it first:\n"
            f"    ros2 run xacro xacro {path.name} > robot.urdf\n"
            f"then point BreakPoint at robot.urdf."
        )

    try:
        tree = ET.ElementTree(ET.fromstring(text))
    except ET.ParseError as e:
        raise URDFImportError(f"{path.name} is not well-formed XML: {e}") from e
    robot = tree.getroot()
    if robot.tag != "robot":
        raise URDFImportError(
            f"{path.name} root element is <{robot.tag}>, expected <robot>; this is not a URDF"
        )

    warnings_: list[str] = []
    unsupported: list[str] = []

    # ---- links + masses ---------------------------------------------------- #
    links = {}
    for link in robot.findall("link"):
        name = link.get("name")
        if not name:
            raise URDFImportError("a <link> has no name attribute")
        mass = _float(link.find("inertial/mass"), "value", 0.0)
        n_col = len(link.findall("collision"))
        links[name] = {"mass": float(mass or 0.0), "n_collision": n_col}
    if not links:
        raise URDFImportError(f"{path.name} declares no <link> elements")

    total_mass = sum(v["mass"] for v in links.values())
    if total_mass <= 0.0:
        raise URDFImportError(
            f"{path.name} has ZERO total mass: every <link> is missing "
            f"<inertial><mass value=.../>. MuJoCo cannot simulate massless moving bodies, "
            f"and a zero-mass robot would report meaningless dynamics. Add real link "
            f"masses/inertias. Links seen: {sorted(links)}"
        )
    total_collisions = sum(v["n_collision"] for v in links.values())
    if total_collisions == 0:
        raise URDFImportError(
            f"{path.name} has NO <collision> geometry on any link (only <visual>). MuJoCo "
            f"discards URDF visuals by default, so this imports as a model with zero geoms: "
            f"the robot would fall through the floor and never touch anything, and every "
            f"failure metric would be meaningless. Add <collision> geometry."
        )

    # ---- joints ------------------------------------------------------------ #
    joints = []
    children = set()
    mimic_joints = []
    refused = []
    for j in robot.findall("joint"):
        jname = j.get("name")
        jtype = (j.get("type") or "").strip().lower()
        parent = j.find("parent")
        child = j.find("child")
        if child is None or child.get("link") is None:
            raise URDFImportError(f"joint {jname!r} has no <child link=.../>")
        if parent is None or parent.get("link") is None:
            raise URDFImportError(f"joint {jname!r} has no <parent link=.../>")
        children.add(child.get("link"))
        if j.find("mimic") is not None:
            mimic_joints.append(jname)
        if jtype in REFUSED_URDF_JOINT_TYPES:
            refused.append(f"{jname} (type={jtype})")
        lim = j.find("limit")
        joints.append({
            "name": jname,
            "type": jtype,
            "parent": parent.get("link"),
            "child": child.get("link"),
            "lower": _float(lim, "lower"),
            "upper": _float(lim, "upper"),
            "effort": _float(lim, "effort"),
        })

    if mimic_joints:
        raise URDFImportError(
            f"URDF <mimic> joints are NOT supported: {mimic_joints}. MuJoCo's URDF importer "
            f"silently DROPS the mimic constraint (measured), so each mimic joint would "
            f"become an independent actuated dof -- a robot that moves in ways the real one "
            f"physically cannot, with results that look fine and are wrong. Replace the "
            f"mimic coupling with a MuJoCo <equality><joint> constraint in a hand-written "
            f"MJCF, or remove the mimic joints from the description."
        )
    if refused:
        raise URDFImportError(
            f"unsupported URDF joint type(s): {refused}. MuJoCo silently expands a planar "
            f"joint into three separate joints (j -> j_TX / j_TY / j_RZ), which breaks the "
            f"one-dof-per-actuated-joint layout BreakPoint's legged field math requires."
        )

    known = set(ACTUATED_URDF_JOINT_TYPES) | set(PASSIVE_URDF_JOINT_TYPES) | {"floating"}
    for jt in sorted({j["type"] for j in joints} - known):
        unsupported.append(f"joint type {jt!r} (not imported as an actuated dof)")

    preexisting_floating = [j["name"] for j in joints if j["type"] == "floating"]
    continuous = [j["name"] for j in joints if j["type"] == "continuous"]
    if continuous:
        warnings_.append(
            f"{len(continuous)} continuous joint(s) {continuous[:6]} have no position "
            f"limits; their actuator ctrlrange defaults to +/-pi (MuJoCo imports them as "
            f"unlimited hinges)."
        )

    # ---- root link --------------------------------------------------------- #
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise URDFImportError(
            f"expected exactly ONE root link (a link that is never a joint's child), found "
            f"{len(roots)}: {sorted(roots)}. A disconnected or multi-root description "
            f"cannot be given a single floating base."
        )
    root_link = roots[0]

    # ---- ROS-only blocks BreakPoint deliberately ignores -------------------- #
    for tag, why in (
        ("transmission", "URDF <transmission> is not imported by MuJoCo -- this is exactly "
                         "why nu==0 on a raw URDF; BreakPoint synthesizes position "
                         "actuators instead"),
        ("gazebo", "<gazebo> blocks (plugins, sensors, materials) are ignored"),
        ("ros2_control", "<ros2_control> hardware/command interfaces are ignored; the "
                         "synthesized actuators are position-controlled regardless of the "
                         "interface your real controller uses"),
    ):
        n = len(robot.findall(tag))
        if n:
            unsupported.append(f"{n} <{tag}> block(s) ignored: {why}")

    # ---- meshes ------------------------------------------------------------ #
    meshes, missing = [], []
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        resolved = _resolve_mesh(fn, path.parent.resolve(), meshdir)
        if resolved is None:
            missing.append(fn)
        else:
            meshes.append({"declared": fn, "resolved": resolved})
            mesh.set("filename", resolved)   # rewrite in the mutable tree (stage 2)
    if missing:
        searched = [str(path.parent.resolve()), str(path.parent.parent.resolve())]
        if meshdir:
            searched.insert(0, str(Path(meshdir).resolve()))
        raise URDFImportError(
            f"{len(missing)} mesh file(s) referenced by {path.name} could not be found:\n"
            + "\n".join(f"    {m}" for m in missing)
            + f"\nSearched: {searched}\n"
            f"MuJoCo does NOT resolve package:// URIs, and a missing mesh would silently "
            f"import as a robot with no collision geometry for those links. Pass "
            f"meshdir=<the directory your package:// paths are relative to>, or ship the "
            f"mesh files next to the URDF."
        )

    return {
        "tree": tree,
        "path": path,
        "robot_name": robot.get("name") or path.stem,
        "root_link": root_link,
        "links": links,
        "joints": joints,
        "urdf_total_mass_kg": float(total_mass),
        "n_collision_elements": int(total_collisions),
        "meshes": meshes,
        "preexisting_floating": preexisting_floating,
        "unsupported": unsupported,
        "warnings": warnings_,
    }


# ---------------------------------------------------------------------------- #
# Stage 2/3 -- floating-base pre-injection + MJCF emission
# ---------------------------------------------------------------------------- #

def _write_prepared_urdf(audit, *, floating_base, out_dir):
    """Write the audited (mesh-path-rewritten) URDF, adding a floating joint on the root.

    The floating joint is injected at the *URDF* level, not the MJCF level, for the reason
    in the module docstring: a URDF root link with no parent joint is FUSED into worldbody
    at compile time, taking its mass with it, so by the time we have MJCF there is no root
    body left to attach a ``<freejoint/>`` to. Giving the root link a parent (a massless
    helper link) keeps it a real body and compiles to ``jnt_type[0] == mjJNT_FREE``.
    """
    tree = audit["tree"]
    robot = tree.getroot()
    injected = False
    if floating_base and not audit["preexisting_floating"]:
        world = ET.Element("link", {"name": _WORLD_LINK_NAME})
        joint = ET.Element("joint", {"name": f"{audit['root_link']}_floating",
                                     "type": "floating"})
        ET.SubElement(joint, "parent", {"link": _WORLD_LINK_NAME})
        ET.SubElement(joint, "child", {"link": audit["root_link"]})
        robot.insert(0, joint)
        robot.insert(0, world)
        injected = True

    out = Path(out_dir) / f"{audit['path'].stem}_bp.urdf"
    tree.write(out, encoding="unicode", xml_declaration=False)
    return out, injected


def _mjcf_text(urdf_file):
    """Compile a URDF and return its MJCF text.

    Prefers ``MjSpec.to_xml()`` (no process-global state). Falls back to compiling +
    ``mj_saveLastXML``, which reads the *last model compiled in this process* -- a global
    that another thread could clobber, hence the preference order.
    """
    import mujoco

    if hasattr(mujoco, "MjSpec"):
        return mujoco.MjSpec.from_file(str(urdf_file)).to_xml()
    model = mujoco.MjModel.from_xml_path(str(urdf_file))
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        mujoco.mj_saveLastXML(tmp, model)
        return Path(tmp).read_text(encoding="utf-8")
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------- #
# Stage 4 -- inject the missing world into the MJCF
# ---------------------------------------------------------------------------- #

def _ensure(parent, tag):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el


def _fmt(*vals):
    return " ".join(f"{float(v):.9g}" for v in vals)


def _find_body(worldbody, name):
    for body in worldbody.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _joint_specs(mjcf_root, audit):
    """Walk the emitted MJCF and return, IN MODEL ORDER, the actuated joints to drive.

    Reads the compiled MJCF rather than the URDF text so ranges/effort come from what
    MuJoCo actually built (URDF ``effort`` lands on the joint's ``actuatorfrcrange``).
    Returns ``(specs, free_joints)`` where each spec is a dict with name/limited/range/
    forcerange/kind.
    """
    by_urdf_type = {j["name"]: j["type"] for j in audit["joints"]}
    specs, free_joints = [], []
    for body in mjcf_root.find("worldbody").iter("body"):
        for jnt in body.findall("joint"):
            jtype = jnt.get("type", "hinge")
            name = jnt.get("name")
            if jtype == "free":
                free_joints.append(name)
                continue
            if jtype in ("ball",):
                raise URDFImportError(
                    f"joint {name!r} compiled to a MuJoCo 'ball' joint (3 dofs). BreakPoint's "
                    f"legged field math assumes one dof per actuated joint; ball joints are "
                    f"not supported."
                )
            rng = jnt.get("range")
            frc = jnt.get("actuatorfrcrange")
            specs.append({
                "_el": jnt,
                "name": name,
                "kind": "slide" if jtype == "slide" else "hinge",
                "range": [float(v) for v in rng.split()] if rng else None,
                "forcerange": [float(v) for v in frc.split()] if frc else None,
                "urdf_type": by_urdf_type.get(name, "?"),
            })
        for jnt in body.findall("freejoint"):
            free_joints.append(jnt.get("name"))
    return specs, free_joints


def _gain_for(spec, kp, kd, kp_error_rad, kp_error_m):
    """Per-joint (kp, kv, source) -- derived from the URDF effort limit unless overridden."""
    if kp is not None:
        kp_j, source = float(kp), "caller"
    else:
        frc = spec["forcerange"]
        effort = max(abs(frc[0]), abs(frc[1])) if frc else 0.0
        if effort > 0:
            err = kp_error_m if spec["kind"] == "slide" else kp_error_rad
            kp_j = float(np.clip(effort / err, *KP_BOUNDS))
            source = "urdf_effort"
        else:
            kp_j, source = FALLBACK_KP, "fallback"
    kv_j = float(kd) if kd is not None else DEFAULT_KD_RATIO * kp_j
    return kp_j, kv_j, source


def _inject_world(mjcf_text, audit, *, floating_base, kp, kd, kp_error_rad, kp_error_m,
                  floor_friction, option, add_camera, add_light,
                  default_armature, default_frictionloss):
    """Inject actuators + floor + light + camera + solver options into the emitted MJCF.

    Structured ElementTree edits, never string surgery, so the result is well-formed by
    construction and every injection is individually reportable.
    """
    root = ET.fromstring(mjcf_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise URDFImportError("emitted MJCF has no <worldbody>; nothing was imported")

    # --- compiler / option -------------------------------------------------- #
    compiler = _ensure(root, "compiler")
    compiler.set("angle", "radian")
    opt = _ensure(root, "option")
    for k, v in {**DEFAULT_OPTION, **(option or {})}.items():
        opt.set(k, str(v))

    # --- the actuated joints ------------------------------------------------ #
    specs, free_joints = _joint_specs(root, audit)
    if floating_base and not free_joints:
        raise URDFImportError(
            f"expected a free joint on the root body {audit['root_link']!r} after "
            f"floating-base injection, found none. The root link may be fused into the "
            f"worldbody (check that it has an <inertial> block)."
        )
    if not specs:
        raise URDFImportError(
            f"{audit['path'].name} has NO actuated joints (every joint is fixed, or the "
            f"description is a single link). There is nothing for a policy to control, so "
            f"a failure search would be vacuous. Add revolute/continuous/prismatic joints."
        )

    # --- floor + light (prepended so they read first in the XML) ------------- #
    floor = ET.Element("geom", {
        "name": FLOOR_GEOM_NAME,
        "type": "plane",
        "size": "0 0 0.05",
        "pos": "0 0 0",
        "friction": _fmt(*floor_friction),
        "condim": "3",
        "rgba": "0.85 0.85 0.87 1",
    })
    worldbody.insert(0, floor)

    # Make the FLOOR the sole author of contact friction.
    #
    # MuJoCo combines a contact's friction as the element-wise MAX of the two geoms.
    # A URDF carries no friction at all, so every robot geom compiles with MuJoCo's
    # default 1.0 -- which beats the injected floor (0.8 nominal) and every value the
    # `friction_floor` theta dim sweeps below it. The dim then appears in every report
    # and atlas as a searched axis while changing nothing: measured, mu=0.18 scored
    # identically to nominal in BOTH engines.
    #
    # Zeroing the robot side makes max(0, mu_floor) == mu_floor, so the swept value is
    # the value the contact actually uses. The cost is that robot-on-robot self
    # contacts become frictionless, which is the right trade for a friction-sweep rig:
    # the floor is the surface being modelled, and a silently dead search axis is far
    # worse than idealised self-contact.
    zeroed_geoms = 0
    for body in worldbody.iter("body"):
        for geom in body.iter("geom"):
            geom.set("friction", "0 0 0")
            zeroed_geoms += 1

    injected_light = False
    if add_light and worldbody.find("light") is None:
        worldbody.insert(1, ET.Element("light", {
            "name": "bp_light", "pos": "0 0 3", "dir": "0 0 -1", "directional": "true",
        }))
        injected_light = True

    # --- tracking camera on the root body ----------------------------------- #
    camera_body = None
    injected_camera = False
    if add_camera:
        camera_body = _find_body(worldbody, audit["root_link"])
        if camera_body is None:
            # Fixed-base robots fuse their root link into worldbody: track the first real
            # body instead, so the renderer still follows the moving part.
            first = next(iter(worldbody.iter("body")), None)
            camera_body = first
        if camera_body is not None and not any(
            c.get("name") == TRACK_CAMERA_NAME for c in camera_body.findall("camera")
        ):
            camera_body.append(ET.Element("camera", {
                "name": TRACK_CAMERA_NAME,
                "mode": "trackcom",
                "pos": "0 -2.0 0.6",
                "xyaxes": "1 0 0 0 0.4 1",
            }))
            injected_camera = True

    # --- one position actuator per actuated joint --------------------------- #
    actuator = _ensure(root, "actuator")
    gains = []
    for spec in specs:
        # URDF has NO armature concept at all, and rotor inertia / joint dry friction are
        # what the `joint_armature_scale` / `joint_frictionloss_scale` theta dims scale.
        # Setting them is opt-in (inventing inertia silently would change the physics), and
        # load_urdf_model warns loudly when they are left at zero -- see "dead_theta_dims".
        el = spec.pop("_el")
        if default_armature is not None:
            el.set("armature", f"{float(default_armature):.6g}")
        if default_frictionloss is not None:
            el.set("frictionloss", f"{float(default_frictionloss):.6g}")
        kp_j, kv_j, source = _gain_for(spec, kp, kd, kp_error_rad, kp_error_m)
        rng = spec["range"]
        if rng is None or (rng[0] == 0.0 and rng[1] == 0.0):
            # Unlimited hinge (URDF `continuous`) or an unlimited slide: no limit to copy.
            rng = [-math.pi, math.pi] if spec["kind"] == "hinge" else [-1.0, 1.0]
            spec["ctrlrange_source"] = "default (joint is unlimited)"
        else:
            spec["ctrlrange_source"] = "urdf joint limit"
        attrs = {
            "name": f"bp_{spec['name']}",
            "joint": spec["name"],
            "kp": f"{kp_j:.6g}",
            "kv": f"{kv_j:.6g}",
            "ctrlrange": _fmt(*rng),
        }
        if spec["forcerange"]:
            attrs["forcerange"] = _fmt(*spec["forcerange"])
        ET.SubElement(actuator, "position", attrs)
        spec["kp"] = kp_j
        spec["kv"] = kv_j
        spec["kp_source"] = source
        spec["ctrlrange"] = [float(rng[0]), float(rng[1])]
        gains.append(kp_j)

    info = {
        "joint_specs": specs,
        "free_joints": free_joints,
        "injected": {
            "freejoint": bool(floating_base and free_joints),
            "freejoint_name": free_joints[0] if free_joints else None,
            "actuators": len(specs),
            "floor_geom": FLOOR_GEOM_NAME,
            "floor_friction": [float(v) for v in floor_friction],
            "camera": TRACK_CAMERA_NAME if injected_camera else None,
            "light": injected_light,
            "option": {**DEFAULT_OPTION, **(option or {})},
        },
        "actuator_kp": {
            "min": float(min(gains)),
            "max": float(max(gains)),
            "sources": sorted({s["kp_source"] for s in specs}),
        },
    }
    return ET.tostring(root, encoding="unicode"), info


# ---------------------------------------------------------------------------- #
# Public: stages 1-4
# ---------------------------------------------------------------------------- #

def urdf_to_mjcf(
    urdf_path,
    *,
    floating_base=True,
    kp=None,
    kd=None,
    kp_error_rad=DEFAULT_KP_ERROR_RAD,
    kp_error_m=DEFAULT_KP_ERROR_M,
    floor_friction=DEFAULT_FLOOR_FRICTION,
    meshdir=None,
    option=None,
    add_camera=True,
    add_light=True,
    default_armature=None,
    default_frictionloss=None,
    work_dir=None,
):
    """URDF -> simulatable MJCF text (stages 1-4). Pure XML + one mujoco compile; no jax.

    Returns ``(mjcf_text, info)``. ``info`` carries the audit, the per-joint actuator specs
    (with the kp/ctrlrange/forcerange actually written and where each came from), and an
    ``injected`` block naming exactly what was added. See ``load_urdf_model`` for the
    parameters -- they are the same, minus the settling measurement.

    Raises ``URDFImportError`` (never a silently-degraded model) on: xacro templates,
    missing/unresolvable meshes, ``<mimic>`` joints, planar joints, ball joints, multiple
    or zero root links, zero total mass, no collision geometry, and no actuated joints.
    """
    audit = _audit_urdf(urdf_path, meshdir=meshdir)
    tmp = work_dir or tempfile.mkdtemp(prefix="bp_urdf_")
    prepared, injected_free = _write_prepared_urdf(
        audit, floating_base=floating_base, out_dir=tmp
    )
    mjcf = _mjcf_text(prepared)
    mjcf, inj = _inject_world(
        mjcf, audit,
        floating_base=floating_base, kp=kp, kd=kd,
        kp_error_rad=kp_error_rad, kp_error_m=kp_error_m,
        floor_friction=floor_friction, option=option,
        add_camera=add_camera, add_light=add_light,
        default_armature=default_armature, default_frictionloss=default_frictionloss,
    )
    info = {
        "urdf_path": str(Path(urdf_path).resolve()),
        "robot_name": audit["robot_name"],
        "root_body": audit["root_link"],
        "floating_base": bool(floating_base),
        "urdf_total_mass_kg": audit["urdf_total_mass_kg"],
        "n_links": len(audit["links"]),
        "n_urdf_joints": len(audit["joints"]),
        "meshes": audit["meshes"],
        "meshdir": str(meshdir) if meshdir else None,
        "unsupported": list(audit["unsupported"]),
        "warnings": list(audit["warnings"]),
        "prepared_urdf_path": str(prepared),
        "work_dir": str(tmp),
        **inj,
    }
    info["injected"]["freejoint"] = bool(injected_free) or info["injected"]["freejoint"]
    info["injected"]["freejoint_source"] = (
        "injected by BreakPoint (URDF-level floating joint)" if injected_free
        else ("already present in the URDF" if audit["preexisting_floating"] else "none")
    )
    return mjcf, info


# ---------------------------------------------------------------------------- #
# Stage 5 -- compile, validate, measure
# ---------------------------------------------------------------------------- #

def _layout(mj_model, info):
    """Resolve + VALIDATE the qpos/qvel layout the observation and field math slice on.

    Fails loud if the actuated joints are not a contiguous block after the free base --
    the invariant ``validate_legged_model`` also enforces, checked here (at import time,
    where we can name the offending joint) rather than at search time.
    """
    import mujoco

    floating = bool(info["floating_base"] and info["free_joints"])
    q0 = 7 if floating else 0
    v0 = 6 if floating else 0
    names = [s["name"] for s in info["joint_specs"]]
    ids = []
    for n in names:
        jid = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n))
        if jid < 0:
            raise URDFImportError(f"joint {n!r} vanished during recompile")
        ids.append(jid)
    qadr = [int(mj_model.jnt_qposadr[i]) for i in ids]
    vadr = [int(mj_model.jnt_dofadr[i]) for i in ids]
    expect_q = list(range(q0, q0 + len(ids)))
    expect_v = list(range(v0, v0 + len(ids)))
    if qadr != expect_q or vadr != expect_v:
        bad = [n for n, a, e in zip(names, qadr, expect_q) if a != e]
        raise URDFImportError(
            f"the actuated joints are not a contiguous block after the base: expected "
            f"qpos addresses {expect_q} / dof addresses {expect_v}, got {qadr} / {vadr} "
            f"(first mismatch at {bad[:3]}). BreakPoint's legged field math slices "
            f"dof[6:] as exactly the actuated joints, so extra passive dofs or a "
            f"non-free-base-first layout would mis-index silently."
        )
    return floating, q0, v0


def _settle_and_measure(mj_model, root_body_id, *, floating, seconds, clearance):
    """Drop the robot holding its default pose and MEASURE where it comes to rest.

    This is the number every derived bound is scaled from (standing height -> theta bounds,
    fall threshold, push magnitude), which is why it is measured rather than copied from
    Go1. Returns a dict with the settled height, the settled uprightness (``R[2,2]``), the
    contact count, and the default pose.
    """
    import mujoco

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    default_pose = np.array(mj_model.qpos0[7 if floating else 0:], dtype=float)
    lo, hi = mj_model.jnt_range[:, 0], mj_model.jnt_range[:, 1]
    limited = mj_model.jnt_limited.astype(bool)
    for k in range(len(default_pose)):
        j = k + (1 if floating else 0)
        if j < mj_model.njnt and limited[j]:
            default_pose[k] = float(np.clip(default_pose[k], lo[j], hi[j]))

    if floating:
        data.qpos[7:] = default_pose
        mujoco.mj_forward(mj_model, data)
        lowest = min(
            (float(data.geom_xpos[g][2] - mj_model.geom_rbound[g])
             for g in range(mj_model.ngeom)
             if mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE),
            default=0.0,
        )
        data.qpos[2] += clearance - lowest
    else:
        data.qpos[:] = default_pose

    if mj_model.nu:
        data.ctrl[:] = default_pose[: mj_model.nu]
    n_steps = max(1, int(round(seconds / float(mj_model.opt.timestep))))
    for _ in range(n_steps):
        mujoco.mj_step(mj_model, data)

    R = np.asarray(data.xmat[root_body_id]).reshape(3, 3)
    height = float(data.qpos[2]) if floating else float(data.xpos[root_body_id][2])
    return {
        "standing_height_m": height,
        "settled_up_proj": float(R[2, 2]),
        "settle_contacts": int(data.ncon),
        "settle_seconds": float(seconds),
        "default_pose": [float(v) for v in default_pose],
        "settle_qvel_norm": float(np.linalg.norm(data.qvel)),
    }


def load_urdf_model(
    urdf_path,
    *,
    floating_base=True,
    kp=None,
    kd=None,
    kp_error_rad=DEFAULT_KP_ERROR_RAD,
    kp_error_m=DEFAULT_KP_ERROR_M,
    floor_friction=DEFAULT_FLOOR_FRICTION,
    meshdir=None,
    option=None,
    add_camera=True,
    add_light=True,
    default_armature=None,
    default_frictionloss=None,
    settle_seconds=1.5,
    clearance=0.02,
    work_dir=None,
):
    """URDF -> a simulatable ``mujoco.MjModel`` + a truthful ``report`` dict.

    Parameters
    ----------
    urdf_path : str | Path
        A FLAT ``.urdf`` (expand xacro first -- a template is rejected, not guessed at).
    floating_base : bool
        ``True`` (default) gives the root link a 6-dof free base -- required for a legged
        robot and for ``make_urdf_target``. ``False`` leaves the root bolted to the world,
        which is what an arm on a table wants (and which MuJoCo does natively by fusing the
        root link into worldbody).
    kp, kd : float | None
        Position-actuator gains applied to EVERY joint. ``None`` (default) derives kp per
        joint from that joint's URDF ``effort`` limit as ``effort / kp_error_*`` and sets
        ``kd = 0.05 * kp``; the report records the resulting range and where it came from.
    kp_error_rad, kp_error_m : float
        The tracking error the derived gain is sized to hold at full effort, for hinge and
        slide joints respectively.
    floor_friction : (mu_slide, mu_torsion, mu_roll)
        Friction of the injected ground plane. ``mu_slide`` is the nominal the
        ``friction_floor`` theta dim perturbs.
    meshdir : str | Path | None
        Directory ``package://<pkg>/<rel>`` paths are resolved against. Meshes are searched
        in ``meshdir``, the URDF's directory, and its parent; an unresolvable mesh RAISES
        with every missing path listed (MuJoCo does not resolve ``package://`` at all).
    option : dict | None
        Overrides merged over ``DEFAULT_OPTION`` (timestep / solver / iterations / ...).
    default_armature, default_frictionloss : float | None
        Rotor armature (reflected inertia) and dry friction to write on EVERY actuated
        joint. Both default to ``None`` = leave whatever the URDF gave. This matters more
        than it looks: **URDF has no armature concept at all**, and ``<dynamics
        friction=>`` is usually absent, so both dof vectors import as all-zero -- which
        makes the ``joint_armature_scale`` and ``joint_frictionloss_scale`` theta dims
        multiply zero by a number, i.e. do NOTHING while still appearing in every report
        and atlas as if they were being searched. The report lists any such dim under
        ``dead_theta_dims`` and warns; set these to make them live (a typical geared servo
        is armature ~ 0.01 kg.m^2, frictionloss ~ 0.1 N.m). Nothing is invented for you,
        because silently adding inertia would change the physics being reported.
    settle_seconds, clearance : float
        The measurement: the robot is dropped from ``clearance`` above the floor holding
        its default pose and simulated for ``settle_seconds`` before the standing height is
        read.

    Returns
    -------
    (mj_model, report)
        ``report`` is plain JSON-able data describing what was INJECTED, what was INFERRED,
        and what was IGNORED -- it is meant to be shown to the user verbatim.

    Raises
    ------
    URDFImportError
        On any description that would compile into a physically meaningless model. See
        ``urdf_to_mjcf``. Additionally, after compiling: a mismatch between the actuator
        count and the actuated-joint count, a non-finite or non-positive settled height, or
        a robot that lost all its mass in the import.
    """
    import mujoco

    mjcf, info = urdf_to_mjcf(
        urdf_path, floating_base=floating_base, kp=kp, kd=kd,
        kp_error_rad=kp_error_rad, kp_error_m=kp_error_m,
        floor_friction=floor_friction, meshdir=meshdir, option=option,
        add_camera=add_camera, add_light=add_light,
        default_armature=default_armature, default_frictionloss=default_frictionloss,
        work_dir=work_dir,
    )
    mjcf_path = Path(info["work_dir"]) / f"{Path(urdf_path).stem}_bp.xml"
    mjcf_path.write_text(mjcf, encoding="utf-8")
    try:
        mj_model = mujoco.MjModel.from_xml_string(mjcf)
    except Exception as e:
        raise URDFImportError(
            f"the injected MJCF failed to compile: {e}\nThe generated file is kept at "
            f"{mjcf_path} for inspection."
        ) from e

    n_expected = len(info["joint_specs"])
    if int(mj_model.nu) != n_expected:
        raise URDFImportError(
            f"injected {n_expected} actuators but the compiled model has nu={mj_model.nu}"
        )
    floating, q0, v0 = _layout(mj_model, info)

    floor_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME))
    root_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, info["root_body"]))
    if floor_id < 0:
        raise URDFImportError("the injected floor geom is missing from the compiled model")
    if root_id < 0:
        if floating:
            raise URDFImportError(
                f"root body {info['root_body']!r} is missing from the compiled model"
            )
        root_id = 1 if mj_model.nbody > 1 else 0   # fixed base: root link fused into world

    model_mass = float(mj_model.body_mass.sum())
    if model_mass <= 0:
        raise URDFImportError("the compiled model has zero total mass")
    n_collision = int(
        sum(1 for g in range(mj_model.ngeom)
            if mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and (mj_model.geom_contype[g] or mj_model.geom_conaffinity[g]))
    )
    if n_collision == 0:
        raise URDFImportError(
            "the compiled model has no collision geometry besides the injected floor"
        )

    measured = _settle_and_measure(
        mj_model, root_id, floating=floating, seconds=settle_seconds, clearance=clearance
    )
    h = measured["standing_height_m"]
    # "Fell through the floor" == the root origin ended up further below ground than the
    # robot's own bounding radius. A root origin that settles at ~0 is legitimate (a
    # wheeled base whose frame sits at ground level), so it warns rather than raising --
    # but derive_theta_space refuses it, since stance-scaled bounds would be nonsense.
    extent = float(max(
        [mj_model.geom_rbound[g] for g in range(mj_model.ngeom)
         if mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE] or [0.0]
    ))
    if not math.isfinite(h) or (floating and h < -extent):
        raise URDFImportError(
            f"settling the robot under gravity produced a non-physical standing height "
            f"({h}): it fell through the floor or diverged. Check the collision geometry "
            f"and the actuator gains (kp range "
            f"{info['actuator_kp']['min']:.4g}..{info['actuator_kp']['max']:.4g})."
        )

    warnings_ = list(info["warnings"])
    if floating and abs(model_mass - info["urdf_total_mass_kg"]) > 1e-6 * max(1.0, model_mass):
        warnings_.append(
            f"compiled mass {model_mass:.6g} kg != URDF mass "
            f"{info['urdf_total_mass_kg']:.6g} kg"
        )
    if not floating:
        warnings_.append(
            f"fixed base: MuJoCo fuses the root link {info['root_body']!r} into the "
            f"worldbody, so its mass is (correctly) absent from the model total "
            f"{model_mass:.6g} kg vs the URDF's {info['urdf_total_mass_kg']:.6g} kg, and "
            f"'standing_height_m' is just body {root_id}'s height, NOT a stance height. "
            f"make_urdf_target refuses fixed-base robots for this reason."
        )
    if measured["settled_up_proj"] < 0.7:
        warnings_.append(
            f"the robot did NOT stay upright while settling (up_proj="
            f"{measured['settled_up_proj']:.3f}). Its default pose is not a standing pose, "
            f"so the derived standing height / fall threshold describe a collapsed robot. "
            f"Give the URDF joint limits that bracket a standing pose, or pass a larger kp."
        )
    if floating and h <= 0.0:
        warnings_.append(
            f"the root link settled at z={h:.4g} m, i.e. at or below the floor. That is "
            f"legitimate for a base frame defined at ground level, but there is no stance "
            f"height to scale a theta space from -- derive_theta_space will refuse it. "
            f"Give the root link an origin above its lowest collision geometry."
        )
        report_stance_ok = False
    else:
        report_stance_ok = True
    if measured["settle_contacts"] == 0 and floating:
        warnings_.append(
            "the robot has no contacts after settling -- it is not touching the floor"
        )

    # A theta dim that multiplies an all-zero model vector is a NO-OP that still shows up
    # in every report/atlas as a searched axis. URDF has no armature concept, and
    # <dynamics friction=> is usually absent, so both are commonly dead here -- say so.
    arm = np.asarray(mj_model.dof_armature)[v0: v0 + mj_model.nu]
    fl = np.asarray(mj_model.dof_frictionloss)[v0: v0 + mj_model.nu]
    dead = []
    if arm.size and not np.any(arm):
        dead.append("joint_armature_scale")
    if fl.size and not np.any(fl):
        dead.append("joint_frictionloss_scale")
    if dead:
        warnings_.append(
            f"theta dim(s) {dead} are NO-OPS on this robot: the corresponding model "
            f"vector is all zeros (URDF has no armature concept; joint dry friction comes "
            f"only from <dynamics friction=...>), so scaling it changes nothing while it "
            f"still appears in reports and the failure atlas as a searched axis. Pass "
            f"default_armature=/default_frictionloss= to load_urdf_model to make them "
            f"live, or read those axes as inactive."
        )

    n_hinge = sum(1 for s in info["joint_specs"] if s["kind"] == "hinge")
    obs_layout = _obs_layout_spec(len(info["joint_specs"]), floating)
    report = {
        **{k: v for k, v in info.items() if k != "tree"},
        "mjcf_path": str(mjcf_path),
        "model_total_mass_kg": model_mass,
        "base_mass_kg": float(mj_model.body_mass[root_id]),
        "n_actuated_joints": n_expected,
        "actuated_joint_names": [s["name"] for s in info["joint_specs"]],
        "joint_kinds": {"hinge": n_hinge, "slide": n_expected - n_hinge},
        "n_collision_geoms": n_collision,
        "nq": int(mj_model.nq), "nv": int(mj_model.nv), "nu": int(mj_model.nu),
        "nbody": int(mj_model.nbody), "ngeom": int(mj_model.ngeom),
        "timestep_s": float(mj_model.opt.timestep),
        "floor_geom_id": floor_id,
        "root_body_id": root_id,
        "qpos_joint_start": q0,
        "qvel_joint_start": v0,
        # False for a fixed base: the settled number is then the root BODY's height, not a
        # stance height, and no theta space / fall threshold may be derived from it.
        "standing_height_is_stance": bool(floating and report_stance_ok),
        "obs_layout": [(n, int(a), int(b)) for n, a, b in obs_layout],
        "obs_dim": int(obs_layout[-1][2]),
        "act_dim": n_expected,
        "obs_convention_warning": OBS_CONVENTION_WARNING,
        "dead_theta_dims": dead,
        "dof_armature_range": [float(arm.min()), float(arm.max())] if arm.size else None,
        "dof_frictionloss_range": [float(fl.min()), float(fl.max())] if fl.size else None,
        "warnings": warnings_,
        **measured,
    }
    return mj_model, report


def format_report(report, *, width=88):
    """Render a ``load_urdf_model`` report as human-readable lines (for the CLI / web)."""
    L = [f"URDF import: {report['robot_name']}  ({report['urdf_path']})"]
    L.append(f"  root body        : {report['root_body']}  "
             f"(floating base: {report['floating_base']})")
    L.append(f"  links / joints   : {report['n_links']} links, "
             f"{report['n_actuated_joints']} actuated joints "
             f"({report['joint_kinds']['hinge']} hinge, "
             f"{report['joint_kinds']['slide']} slide)")
    L.append(f"  mass             : {report['model_total_mass_kg']:.4g} kg total "
             f"(base {report['base_mass_kg']:.4g} kg); URDF declared "
             f"{report['urdf_total_mass_kg']:.4g} kg")
    L.append(f"  collision geoms  : {report['n_collision_geoms']}"
             f"   meshes: {len(report['meshes'])}")
    inj = report["injected"]
    L.append(f"  INJECTED         : freejoint={inj['freejoint']} "
             f"({inj['freejoint_source']}), {inj['actuators']} position actuators "
             f"(kp {report['actuator_kp']['min']:.4g}..{report['actuator_kp']['max']:.4g}, "
             f"from {'/'.join(report['actuator_kp']['sources'])}), "
             f"floor '{inj['floor_geom']}' friction {inj['floor_friction'][0]}, "
             f"camera={inj['camera']}, light={inj['light']}")
    label = "standing height" if report["standing_height_is_stance"] else "root body height"
    L.append(f"  MEASURED         : {label} {report['standing_height_m']:.4g} m "
             f"after {report['settle_seconds']:.3g}s settle "
             f"(up_proj {report['settled_up_proj']:.3f}, "
             f"{report['settle_contacts']} contacts)")
    L.append(f"  obs / act        : {report['obs_dim']} / {report['act_dim']}   layout: "
             + ", ".join(f"{n}[{a}:{b}]" for n, a, b in report["obs_layout"]))
    for u in report["unsupported"]:
        L.append(f"  IGNORED          : {u}")
    for w in report["warnings"]:
        L.append(f"  WARNING          : {w}")
    L.append("  NOTE             : " + OBS_CONVENTION_WARNING)
    return "\n".join(L)


# ---------------------------------------------------------------------------- #
# The observation layout (shared by the report and the env)
# ---------------------------------------------------------------------------- #

def _obs_layout_spec(n_joints, floating):
    """The (name, start, stop) blocks of the synthesized observation. Single source of
    truth for both ``report["obs_layout"]`` and ``URDFEnv.obs_layout``."""
    blocks = []
    if floating:
        blocks += [("gravity_body", 3), ("lin_vel_body", 3), ("ang_vel_body", 3)]
    blocks += [("joint_pos", n_joints), ("joint_vel", n_joints),
               ("last_action", n_joints), ("command", 3)]
    out, i = [], 0
    for name, n in blocks:
        out.append((name, i, i + n))
        i += n
    return out


# ---------------------------------------------------------------------------- #
# The env
# ---------------------------------------------------------------------------- #

class URDFEnv:
    """Minimal MJX env for a URDF-imported robot -- exactly the contract ``rollout_one`` needs.

    Contract implemented (see ``breakpoint_mjx/rollout.py``): ``reset(key) -> State``,
    ``step(state, action) -> State``, ``state.obs`` as the legged-convention dict
    ``{"state": vector}``, ``state.data`` (an ``mjx.Data``), plus ``mjx_model``
    (rebindable via ``env._mjx_model``, which is how the evaluator gives each vmapped row
    its own perturbed model), ``observation_size``, ``action_size``, ``dt`` and ``render``.

    OBSERVATION LAYOUT (a BreakPoint convention -- see the module docstring's honesty note;
    ``obs_layout`` exposes it programmatically). All blocks are RAW SI units, unscaled, in
    this order, concatenated into ``obs["state"]``:

    ==================  ====  ==========================================================
    block               dims  meaning
    ==================  ====  ==========================================================
    ``gravity_body``      3   the gravity direction in the base frame, ``R.T @ [0,0,-1]``
                              (the standard "projected gravity" attitude cue). Floating
                              base only.
    ``lin_vel_body``      3   base linear velocity in the base frame, ``R.T @ qvel[:3]``.
                              Floating base only.
    ``ang_vel_body``      3   base angular velocity in the base frame, ``R.T @ qvel[3:6]``.
                              Floating base only.
    ``joint_pos``         nu  ``qpos[joint block] - default_pose`` (offset from the settled
                              default pose, not absolute angles), in model joint order.
    ``joint_vel``         nu  ``qvel[joint block]``, same order.
    ``last_action``       nu  the previous step's action, as the policy emitted it
                              (post-clip, pre-scaling).
    ``command``           3   the velocity command ``(vx, vy, yaw_rate)`` this episode.
    ==================  ====  ==========================================================

    Total: ``9 + 3*nu + 3`` for a floating base, ``3*nu + 3`` for a fixed base. Joint order
    is the compiled model's joint order, which is URDF declaration order --
    ``report["actuated_joint_names"]`` lists it.

    ACTION: shape ``(nu,)``, clipped to ``[-1, 1]``, applied as
    ``ctrl = default_pose + action_scale * action`` clipped to each actuator's
    ``ctrlrange`` (the URDF joint limits). This is the MuJoCo Playground locomotion
    convention; ``action_scale`` defaults to 0.3 rad as it does there.

    A shared env instance is stateful under jit (``rollout_one`` assigns ``env._mjx_model``),
    so build a FRESH env per evaluator rather than sharing one across jit boundaries.
    """

    def __init__(self, mj_model, report, *, ctrl_dt=0.02, action_scale=0.3,
                 command=(0.5, 0.0, 0.0), command_range=None, qpos_noise=0.0):
        import jax.numpy as jnp
        from mujoco import mjx

        self._mj_model = mj_model
        self.report = dict(report)
        self._mjx_model = mjx.put_model(mj_model)
        self._floating = bool(report["floating_base"] and report["free_joints"])
        self._nu = int(mj_model.nu)
        self._q0 = int(report["qpos_joint_start"])
        self._v0 = int(report["qvel_joint_start"])
        self._root_body_id = int(report["root_body_id"])
        self._action_scale = float(action_scale)
        self._qpos_noise = float(qpos_noise)

        sim_dt = float(mj_model.opt.timestep)
        self._n_substeps = max(1, int(round(float(ctrl_dt) / sim_dt)))
        self._dt = self._n_substeps * sim_dt

        default_pose = np.asarray(report["default_pose"], dtype=np.float64)
        init_qpos = np.asarray(mj_model.qpos0, dtype=np.float64).copy()
        init_qpos[self._q0:] = default_pose
        if self._floating:
            init_qpos[:3] = [0.0, 0.0, float(report["standing_height_m"])]
            init_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._init_qpos = jnp.asarray(init_qpos)
        self._default_pose = jnp.asarray(default_pose)
        self._ctrl_lo = jnp.asarray(mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_hi = jnp.asarray(mj_model.actuator_ctrlrange[:, 1])
        self._command = jnp.asarray(np.asarray(command, dtype=np.float64))
        self._command_range = (
            None if command_range is None
            else np.asarray(command_range, dtype=np.float64).reshape(3, 2)
        )
        self._obs_layout = tuple(
            (n, slice(a, b)) for n, a, b in _obs_layout_spec(self._nu, self._floating)
        )
        self._obs_dim = self._obs_layout[-1][1].stop
        from breakpoint_mjx.obs_adversary import set_obs_non_sensor
        set_obs_non_sensor(self, [(sl.start, sl.stop - sl.start) for n, sl in self._obs_layout
                                  if n in ("last_action", "command")])

    # -- contract surface ---------------------------------------------------- #
    @property
    def mjx_model(self):
        """The MJX model. ``rollout_one`` REBINDS ``env._mjx_model`` per vmapped row."""
        return self._mjx_model

    @property
    def mj_model(self):
        """The host-side ``MjModel`` (what ``resolve_constants_by_name`` reads)."""
        return self._mj_model

    @property
    def action_size(self):
        return self._nu

    @property
    def observation_size(self):
        """Length of ``obs["state"]``. (An int, not the brax dict-of-shapes form.)"""
        return self._obs_dim

    @property
    def obs_layout(self):
        """``((name, slice), ...)`` for ``obs["state"]`` -- build your ``ObsSpec`` from this."""
        return self._obs_layout

    @property
    def dt(self):
        """Control timestep in seconds (``n_substeps * opt.timestep``)."""
        return self._dt

    # -- physics ------------------------------------------------------------- #
    def _obs(self, data, info):
        import jax.numpy as jnp

        parts = []
        if self._floating:
            R = data.xmat[self._root_body_id].reshape(3, 3)
            parts += [
                R.T @ jnp.array([0.0, 0.0, -1.0]),
                R.T @ data.qvel[:3],
                R.T @ data.qvel[3:6],
            ]
        parts += [
            data.qpos[self._q0: self._q0 + self._nu] - self._default_pose,
            data.qvel[self._v0: self._v0 + self._nu],
            info["last_act"],
            info["command"],
        ]
        return jnp.concatenate(parts)

    def reset(self, rng):
        """Reset to the settled default pose. ``rng`` samples the velocity command when
        ``command_range`` was given, and the optional joint-position jitter; with both at
        their defaults the reset is deterministic (perturbation comes from the theta space,
        not from a hidden reset distribution)."""
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        rng, k_cmd, k_q = jax.random.split(rng, 3)
        qpos = self._init_qpos
        if self._qpos_noise > 0:
            jitter = self._qpos_noise * jax.random.normal(k_q, (self._nu,))
            qpos = qpos.at[self._q0: self._q0 + self._nu].add(jitter)
        if self._command_range is None:
            command = self._command
        else:
            lo = jnp.asarray(self._command_range[:, 0])
            hi = jnp.asarray(self._command_range[:, 1])
            command = jax.random.uniform(k_cmd, (3,), minval=lo, maxval=hi)

        data = mjx.make_data(self._mjx_model)
        data = data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(data.qvel),
            ctrl=jnp.asarray(self._default_pose[: self._nu]),
        )
        data = mjx.forward(self._mjx_model, data)
        info = {
            "command": command,
            "last_act": jnp.zeros((self._nu,)),
            "step": jnp.array(0, dtype=jnp.int32),
            "rng": rng,
        }
        return _URDFState(
            data=data, obs={"state": self._obs(data, info)},
            reward=jnp.zeros(()), done=jnp.zeros(()), metrics={}, info=info,
        )

    def step(self, state, action):
        """Advance ``n_substeps`` physics steps under ``ctrl = default_pose + scale*action``."""
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        act = jnp.clip(action, -1.0, 1.0)
        ctrl = jnp.clip(self._default_pose + self._action_scale * act,
                        self._ctrl_lo, self._ctrl_hi)
        data = state.data.replace(ctrl=ctrl)
        model = self._mjx_model
        data = jax.lax.fori_loop(
            0, self._n_substeps, lambda _, d: mjx.step(model, d), data
        )
        info = {**state.info, "last_act": act, "step": state.info["step"] + 1}
        return state.replace(
            data=data, obs={"state": self._obs(data, info)}, info=info
        )

    # -- rendering (for breakpoint_mjx.render.render_failure) ---------------- #
    def render(self, states, camera=TRACK_CAMERA_NAME, height=240, width=320):
        """Render a list of States to RGB frames on the HOST model (numpy, not jax)."""
        import mujoco

        data = mujoco.MjData(self._mj_model)
        frames = []
        with mujoco.Renderer(self._mj_model, height=height, width=width) as renderer:
            for st in states:
                data.qpos[:] = np.asarray(st.data.qpos, dtype=np.float64)
                data.qvel[:] = np.asarray(st.data.qvel, dtype=np.float64)
                mujoco.mj_forward(self._mj_model, data)
                renderer.update_scene(data, camera=(-1 if camera is None else camera))
                frames.append(renderer.render())
        return frames


def _make_state_class():
    """Define the flax struct State lazily (flax is only present where jax is)."""
    from typing import Any, Dict

    import jax
    from flax import struct
    from mujoco import mjx

    @struct.dataclass
    class _State:
        """Same shape as ``mujoco_playground``'s ``State`` (data/obs/reward/done/metrics/
        info), defined locally so this module does not require mujoco_playground."""

        data: mjx.Data
        obs: Any
        reward: jax.Array
        done: jax.Array
        metrics: Dict[str, jax.Array]
        info: Dict[str, Any]

    return _State


class _LazyState:
    """Module-level stand-in so ``_URDFState`` can be referenced before jax is imported."""

    _cls = None

    def __call__(self, **kwargs):
        if _LazyState._cls is None:
            _LazyState._cls = _make_state_class()
        return _LazyState._cls(**kwargs)


_URDFState = _LazyState()


# ---------------------------------------------------------------------------- #
# Derived theta space
# ---------------------------------------------------------------------------- #

# Dimensionless dims -- scale factors and probabilities that transfer across robots
# unchanged, copied from configs/legged_default/theta_space.yaml. Only the DIMENSIONAL
# dims below are re-derived from the measured robot.
_SCALE_FREE_DIMS = {
    "kp_scale": ("model", (0.50, 1.50), 1.0, (0.85, 1.15)),
    "kd_scale": ("model", (0.50, 1.50), 1.0, (0.85, 1.15)),
    "joint_frictionloss_scale": ("model", (0.5, 3.0), 1.0, (0.9, 1.3)),
    "joint_armature_scale": ("model", (0.8, 2.0), 1.0, (1.0, 1.1)),
    "floor_slope_rad": ("model", (-0.20, 0.20), 0.0, (-0.10, 0.10)),
    "obs_noise_scale": ("rollout", (0.0, 0.10), 0.0, (0.0, 0.04)),
    "latency_steps": ("rollout", (0.0, 4.0), 0.0, (0.0, 2.0)),
    "obs_dropout_prob": ("rollout", (0.0, 0.50), 0.0, (0.0, 0.20)),
    "obs_bias_scale": ("rollout", (0.0, 0.10), 0.0, (0.0, 0.04)),
    "push_step_frac": ("rollout", (0.0, 1.0), 0.0, (0.0, 1.0)),
}


def derive_theta_space(report, *, mass_frac=(-0.08, 0.47), mass_plausible=(-0.08, 0.16),
                       com_xy_frac=0.28, com_z_frac=0.18, push_froude=1.2):
    """Build a ``ThetaSpace`` whose DIMENSIONAL bounds are scaled from THIS robot.

    Copying Go1's ``theta_space.yaml`` onto an arbitrary URDF is the silent-wrongness this
    avoids: ``torso_mass_add_kg`` of +6 kg is a 47% payload on a 12.7 kg Go1 and a 4x
    payload on a 1.5 kg toy, and Go1's 0.18 m fall threshold never triggers on a robot that
    stands at 0.8 m. So:

    * ``friction_floor``    -- scaled around the injected floor's actual mu.
    * ``torso_mass_add_kg`` -- a FRACTION of the robot's real total mass
                               (default ``-8%`` .. ``+47%``, the Go1 ratios).
    * ``com_offset_{x,y,z}_m`` -- a fraction of the MEASURED standing height
                               (0.28 / 0.18, again the Go1 ratios at Go1's 0.28 m stance).
    * ``push_mag``          -- Froude-scaled: ``push_froude * sqrt(g * standing_height)``,
                               which reproduces Go1's 2.0 m/s at its 0.28 m stance.

    The dimensionless dims (gain scales, armature/friction scales, floor slope, and every
    observation/latency adversary) transfer unchanged and are copied verbatim from
    ``configs/legged_default/theta_space.yaml``.

    Dim NAMES and ORDER match what ``build_legged_fields`` / ``rollout_one`` read by name.
    """
    mass = float(report["model_total_mass_kg"])
    h = float(report["standing_height_m"])
    mu = float(report["injected"]["floor_friction"][0])
    if not (mass > 0 and h > 0):
        raise URDFImportError(
            f"cannot derive a theta space from mass={mass} standing_height={h}"
        )
    if not report.get("standing_height_is_stance", True):
        raise URDFImportError(
            "this report came from a FIXED-base import, whose measured height is the root "
            "body's height, not a stance height. Scaling payload / CoM / push bounds from "
            "it would produce meaningless numbers; import with floating_base=True."
        )
    g = 9.81
    push_max = push_froude * math.sqrt(g * h)

    dims: dict[str, ThetaDim] = {}

    def add(name, kind, bounds, nominal, plausible):
        dims[name] = ThetaDim(name=name, kind=kind,
                              bounds=(float(bounds[0]), float(bounds[1])),
                              nominal=float(nominal),
                              plausible=(float(plausible[0]), float(plausible[1])))

    # Order mirrors configs/legged_default/theta_space.yaml so a derived space reads the
    # same way in reports/atlases as a hand-written one.
    add("friction_floor", "model", (0.1875 * mu, 1.5 * mu), mu, (0.5 * mu, 1.25 * mu))
    add("torso_mass_add_kg", "model", (mass_frac[0] * mass, mass_frac[1] * mass), 0.0,
        (mass_plausible[0] * mass, mass_plausible[1] * mass))
    add("com_offset_x_m", "model", (-com_xy_frac * h, com_xy_frac * h), 0.0,
        (-0.625 * com_xy_frac * h, 0.625 * com_xy_frac * h))
    add("com_offset_y_m", "model", (-com_xy_frac * h, com_xy_frac * h), 0.0,
        (-0.625 * com_xy_frac * h, 0.625 * com_xy_frac * h))
    add("kp_scale", *_SCALE_FREE_DIMS["kp_scale"])
    add("kd_scale", *_SCALE_FREE_DIMS["kd_scale"])
    add("com_offset_z_m", "model", (-com_z_frac * h, com_z_frac * h), 0.0,
        (-0.6 * com_z_frac * h, 0.6 * com_z_frac * h))
    add("joint_frictionloss_scale", *_SCALE_FREE_DIMS["joint_frictionloss_scale"])
    add("joint_armature_scale", *_SCALE_FREE_DIMS["joint_armature_scale"])
    add("obs_noise_scale", *_SCALE_FREE_DIMS["obs_noise_scale"])
    add("latency_steps", *_SCALE_FREE_DIMS["latency_steps"])
    add("obs_dropout_prob", *_SCALE_FREE_DIMS["obs_dropout_prob"])
    add("obs_bias_scale", *_SCALE_FREE_DIMS["obs_bias_scale"])
    add("push_mag", "rollout", (0.0, push_max), 0.0, (0.0, 0.5 * push_max))
    add("push_step_frac", *_SCALE_FREE_DIMS["push_step_frac"])
    add("floor_slope_rad", *_SCALE_FREE_DIMS["floor_slope_rad"])
    return ThetaSpace(dims=dims)


def derive_thresholds(report, *, fall_frac=0.5):
    """Per-robot failure-threshold overrides for ``Target.thresholds``.

    Only ``trunk_height_min`` is robot-scale-dependent: it is ``fall_frac`` of the MEASURED
    standing height (the rule ``targets/g1.py`` states -- ~half the nominal base height).
    ``up_proj_min`` / ``ang_vel_max`` / ``tracking_err_max`` are scale-free and left to
    ``configs/search.yaml``.
    """
    return {"trunk_height_min": round(fall_frac * float(report["standing_height_m"]), 6)}


def write_theta_space_yaml(space, path, *, header=None):
    """Write a ``ThetaSpace`` to a ``theta_space.yaml`` that ``config.load_theta_space``
    reads back verbatim. Returns the resolved path."""
    import yaml

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "dims": {
            d.name: {
                "kind": d.kind,
                "bounds": [round(float(d.bounds[0]), 6), round(float(d.bounds[1]), 6)],
                "nominal": round(float(d.nominal), 6),
                "plausible": [round(float(d.plausible[0]), 6),
                              round(float(d.plausible[1]), 6)],
            }
            for d in space.dims.values()
        }
    }
    text = yaml.safe_dump(body, sort_keys=False, default_flow_style=None)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- #
# The Target
# ---------------------------------------------------------------------------- #

def make_urdf_target(name, urdf_path, *, theta_space_path=None, checkpoint_name=None,
                     env_kwargs=None, fall_frac=0.5, space_kwargs=None, **load_kwargs):
    """Build a real ``Target`` from a URDF -- the entry point for bring-your-own-robot.

    The URDF is imported ONCE here (mujoco only, no jax) so the theta space and fall
    threshold can be derived from the measured robot; the returned ``Target.load_env``
    builds the MJX env lazily, so a caller can construct the target on a CPU box and only
    pay for jax when the pipeline runs.

    Parameters
    ----------
    name : str
        Registry-style name; also the default checkpoint stem (``<name>_policy.pkl``) and
        the derived config directory.
    urdf_path : str | Path
        The flat ``.urdf``.
    theta_space_path : str | Path | None
        Where to WRITE the derived ``theta_space.yaml``. ``None`` writes it next to the
        URDF as ``<urdf dir>/bp_<name>_theta_space.yaml``. The path is stored ABSOLUTE on
        the Target, which is what the pipeline's ``config_path(target.theta_space_path)`` join
        resolves to (pathlib returns the absolute right-hand operand).
    env_kwargs : dict | None
        Forwarded to ``URDFEnv`` (``ctrl_dt``, ``action_scale``, ``command``,
        ``command_range``, ``qpos_noise``).
    fall_frac : float
        Fraction of the measured standing height used as ``trunk_height_min``.
    space_kwargs : dict | None
        Forwarded to ``derive_theta_space``.
    **load_kwargs
        Forwarded to ``load_urdf_model`` (``kp``, ``meshdir``, ``floor_friction``, ...).

    Returns
    -------
    Target
        Wired to the SHARED legged field math (``build_legged_fields`` /
        ``LEGGED_MODEL_FIELDS``) and the legged objective, with ``camera="track"`` and
        per-robot ``thresholds``. The import ``report`` is attached to ``load_env`` as
        ``target.load_env.report`` (the frozen dataclass has no field for it).

    Raises
    ------
    URDFImportError
        If ``floating_base=False``: BreakPoint's legged objective reads the base pose
        positionally from ``qpos[2]`` / ``qvel[:3]`` and the field math slices ``dof[6:]``,
        both of which require a 6-dof free base. A bolted-down arm needs a manipulation
        objective (``breakpoint_mjx/objectives/manipulation.py``), not this factory --
        ``load_urdf_model(..., floating_base=False)`` still imports it for inspection.
    """
    from breakpoint_mjx.targets import Target
    from breakpoint_mjx.targets._legged import LEGGED_MODEL_FIELDS, build_legged_fields
    from breakpoint_mjx.targets.generic import resolve_constants_from_candidates

    if load_kwargs.get("floating_base", True) is False:
        raise URDFImportError(
            "make_urdf_target requires floating_base=True. The legged objective reads the "
            "base height/velocity positionally (qpos[2], qvel[:3], qvel[3:6]) and the "
            "legged field math slices dof[6:] as the actuated joints, so a fixed-base "
            "robot (an arm) would be scored on meaningless numbers. Use "
            "load_urdf_model(..., floating_base=False) to inspect such a model, and pair "
            "it with a manipulation Objective to search it."
        )

    mj_model, report = load_urdf_model(urdf_path, **load_kwargs)
    space = derive_theta_space(report, **(space_kwargs or {}))

    if theta_space_path is None:
        theta_space_path = Path(report["urdf_path"]).parent / f"bp_{name}_theta_space.yaml"
    ts_path = write_theta_space_yaml(
        space, theta_space_path,
        header=(
            f"DERIVED theta space for URDF target {name!r} "
            f"({Path(report['urdf_path']).name}).\n"
            f"Dimensional bounds are scaled from the MEASURED robot: total mass "
            f"{report['model_total_mass_kg']:.4g} kg, standing height "
            f"{report['standing_height_m']:.4g} m (settled under gravity), floor mu "
            f"{report['injected']['floor_friction'][0]}.\n"
            f"Dimensionless dims are copied from configs/legged_default/theta_space.yaml.\n"
            f"Regenerate with breakpoint_mjx.urdf_target.make_urdf_target; do not "
            f"hand-edit unless you also update the target."
        ),
    ).resolve()

    def load_env():
        env = URDFEnv(mj_model, report, **(env_kwargs or {}))
        return env

    load_env.report = report          # the import report, for the CLI / web report
    load_env.space = space
    load_env.mj_model = mj_model

    def resolve_constants(env):
        from breakpoint_mjx.targets._legged import validate_legged_model

        validate_legged_model(env.mjx_model)
        return resolve_constants_from_candidates(
            env, (FLOOR_GEOM_NAME,), (report["root_body"],)
        )

    return Target(
        name=name,
        env_name=f"urdf:{Path(report['urdf_path']).name}",
        checkpoint_name=checkpoint_name or f"{name}_policy.pkl",
        theta_space_path=str(ts_path),
        model_fields=LEGGED_MODEL_FIELDS,
        load_env=load_env,
        resolve_constants=resolve_constants,
        build_fields=build_legged_fields,
        thresholds=derive_thresholds(report, fall_frac=fall_frac),
        camera=TRACK_CAMERA_NAME,
    )
