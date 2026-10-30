"""The AR4's real link meshes for the 3D viewer, fetched from the driver repo on demand.

The STL files in ``annin_ar4_description/meshes/ar4_mk5`` are ~50 MB, far more than a
browser view needs, so ``fetch`` downloads them once (pinned commit, MIT licence) and
decimates each by vertex clustering to a few thousand triangles. The viewer falls back to
simple shapes when the meshes are not there; nothing in the simulation uses them.
"""
from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO = "https://raw.githubusercontent.com/Annin-Robotics/ar4_ros_driver"
COMMIT = "6a3ebb11cedab12cd1d29b41d63aa270a008ab8b"
MESH_DIR = "annin_ar4_description/meshes/ar4_mk5"

ALU, MOTOR, BLUE, WHITE, ENCL = "#dcdcdc", "#141414", "#0059ff", "#ffffff", "#a0a0a0"
# (file, link, visual origin xyz, rpy, colour), from ar_macro.xacro
PARTS = [
    ("Link_Base_Aluminum", "base", (0, 0, 0), (0, 0, -1.5708), ALU),
    ("Link_Base_Enclosure", "base", (0, 0, 0), (0, 0, -1.5708), ENCL),
    ("Link_Base_Motor", "base", (0, 0, 0), (0, 0, -1.5708), MOTOR),
    ("Link_1_Aluminum", "link_1", (0, 0, 0), (0, 0, 0), ALU),
    ("Link_1_Motor", "link_1", (0, 0, 0), (0, 0, 0), MOTOR),
    ("Link_2_Aluminum", "link_2", (0, 0, -0.00887), (3.14, 0, 0), ALU),
    ("Link_2_Motor", "link_2", (0, 0, -0.00887), (3.14, 0, 0), MOTOR),
    ("Link_2_Cover", "link_2", (0, 0, -0.00887), (3.14, 0, 0), BLUE),
    ("Link_2_Logo", "link_2", (0, 0, -0.00887), (3.14, 0, 0), WHITE),
    ("Link_3_Aluminum", "link_3", (0, 0, -0.03671), (0, 0, 0), ALU),
    ("Link_3_Motor", "link_3", (0, 0, -0.03671), (0, 0, 0), MOTOR),
    ("Link_4_Aluminum", "link_4", (0, 0, -0.07594), (0, 0, -1.5708), ALU),
    ("Link_4_Motor", "link_4", (0, 0, -0.07594), (0, 0, -1.5708), MOTOR),
    ("Link_4_Cover", "link_4", (0, 0, -0.07594), (0, 0, -1.5708), BLUE),
    ("Link_4_Logo", "link_4", (0, 0, -0.07594), (0, 0, -1.5708), WHITE),
    ("Link_5_Aluminum", "link_5", (0, 0, -0.0275), (0, 0, 0), ALU),
    ("Link_5_Motor", "link_5", (0, 0, -0.0275), (0, 0, 0), MOTOR),
    ("Link_6_Aluminum", "link_6", (0, 0, -0.016), (0, 0, 0), ALU),
]


def read_stl(data: bytes) -> np.ndarray:
    """Triangles (N, 3, 3) from binary or ASCII STL bytes."""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                      ("a", "<u2")]), count=n, offset=84)
            return rec["v"].astype(np.float64)
    verts = [list(map(float, ln.split()[1:4])) for ln in data.decode("ascii", "ignore").splitlines()
             if ln.strip().startswith("vertex")]
    return np.asarray(verts, float).reshape(-1, 3, 3)


def decimate(tris: np.ndarray, target: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering on a grid sized so that about ``target`` triangles survive.
    Returns (vertices (V, 3) float32, faces (F, 3) uint32)."""
    v = tris.reshape(-1, 3)
    span = float(np.max(v.max(0) - v.min(0))) or 1.0
    cell = span / 200.0
    for _ in range(12):
        key = np.floor((v - v.min(0)) / cell).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        f = inv.reshape(-1, 3)
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = np.unique(np.sort(f, 1), axis=0, return_index=True)[1]
        if len(f) <= target or cell > span / 8:
            break
        cell *= 1.35
    inv = inv.ravel()
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, v)
    cnt = np.bincount(inv, minlength=len(uniq))[:, None]
    verts = (sums / np.maximum(cnt, 1)).astype(np.float32)
    faces = inv.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    _, idx = np.unique(np.sort(faces, 1), axis=0, return_index=True)
    return verts, faces[np.sort(idx)].astype(np.uint32)


def fetch(out_dir: Path, *, target: int = 4000, progress=print) -> Path:
    """Download and decimate the MK5 meshes into ``out_dir``; writes ``manifest.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": f"github.com/Annin-Robotics/ar4_ros_driver@{COMMIT}/{MESH_DIR}",
                "licence": "MIT (Copyright (c) 2021 Dexter Ong; Annin Robotics)", "parts": []}
    for name, link, xyz, rpy, colour in PARTS:
        url = f"{REPO}/{COMMIT}/{MESH_DIR}/{name}.STL"
        progress(f"fetching {name}.STL")
        with urllib.request.urlopen(url, timeout=60) as r:
            tris = read_stl(r.read())
        verts, faces = decimate(tris, target)
        fn = f"{name}.bin"
        (out_dir / fn).write_bytes(struct.pack("<II", len(verts), len(faces))
                                   + verts.tobytes() + faces.tobytes())
        index = -1 if link == "base" else int(link.split("_")[1]) - 1
        manifest["parts"].append({"file": fn, "link": index, "xyz": xyz, "rpy": rpy,
                                  "color": colour, "triangles": int(len(faces))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir / "manifest.json"
