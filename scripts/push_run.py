"""Push a finished run from <BP_OUT>/<target>/ to BreakPoint Web.

Usage:
  BP_WEB_URL=https://... BP_API_KEY=bp_... python scripts/push_run.py --target go1

Flow: register (JSON) -> PUT each artifact to the returned upload URL -> finalize.
Idempotent: pass --run-id to reuse a UUID and resume a partial push; a finalized
run returns 409. Stdlib only (urllib), so it runs on a bare Colab/CI box.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.error
import urllib.request
import uuid


def _api(url: str, method: str, payload, key: str):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, method=method, data=data,
        headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body or b"{}")
        except Exception:
            return e.code, {"error": body.decode("utf-8", "replace")}


def _put_blob(upload: dict, body: bytes, content_type: str, key: str) -> None:
    # Uploads go through BreakPoint Web's own auth-checked endpoint (which stores the
    # blob privately), so authenticate with the project API key, same as register.
    req = urllib.request.Request(
        upload["uploadUrl"], method="PUT", data=body,
        headers={"authorization": f"Bearer {key}", "content-type": content_type},
    )
    urllib.request.urlopen(req).read()


def build_payload(out_dir: str, run_id: str, version_label: str) -> dict:
    """Assemble the register payload from <out_dir>/families.json + on-disk artifacts."""
    with open(os.path.join(out_dir, "families.json")) as fh:
        fam = json.load(fh)

    artifacts = [{"name": "failures.json.gz", "kind": "raw", "content_type": "application/gzip"}]
    if os.path.exists(os.path.join(out_dir, "atlas_history")) or os.path.exists(
        os.path.join(out_dir, "atlas.json")
    ):
        if os.path.exists(os.path.join(out_dir, "atlas.json")):
            artifacts.append({"name": "atlas.json", "kind": "atlas", "content_type": "application/json"})
    # Family videos: run_pipeline writes <fam.id>.mp4; families.json carries names, not ids,
    # so map by the family's discovered mp4 if present (fam-<k>.mp4 fallback for legacy).
    for i, f in enumerate(fam.get("families", [])):
        mp4 = _family_mp4(out_dir, f, i)
        if mp4:
            artifacts.append({"name": f"video:{f['name']}", "kind": "video", "content_type": "video/mp4"})

    payload = {
        **fam,
        "run_id": run_id,
        "version_label": version_label,
        "artifacts": artifacts,
    }
    reg = os.path.join(out_dir, "regression.json")
    if os.path.exists(reg):
        with open(reg) as fh:
            payload["regression"] = json.load(fh)
    hard = os.path.join(out_dir, "hardening.json")
    if os.path.exists(hard):
        with open(hard) as fh:
            payload["hardening"] = json.load(fh)
    return payload


def _family_mp4(out_dir: str, family: dict, index: int) -> str | None:
    """Best-effort mp4 path for a family. run_pipeline names them fam-<k>.mp4."""
    for cand in (family.get("id"), f"fam-{index}", family["name"].replace(" ", "_")):
        if not cand:
            continue
        p = os.path.join(out_dir, f"{cand}.mp4")
        if os.path.exists(p):
            return p
    return None


def _artifact_bytes(out_dir: str, name: str, families: list[dict]) -> tuple[bytes, str]:
    if name == "failures.json.gz":
        raw = os.path.join(out_dir, "failures.json")
        data = open(raw, "rb").read() if os.path.exists(raw) else b"{}"
        return gzip.compress(data), "application/gzip"
    if name == "atlas.json":
        return open(os.path.join(out_dir, "atlas.json"), "rb").read(), "application/json"
    if name.startswith("video:"):
        fam_name = name[len("video:") :]
        for i, f in enumerate(families):
            if f["name"] == fam_name:
                mp4 = _family_mp4(out_dir, f, i)
                if mp4:
                    return open(mp4, "rb").read(), "video/mp4"
    raise FileNotFoundError(name)


def push(out_dir: str, web_url: str, key: str, version_label: str, run_id: str) -> int:
    payload = build_payload(out_dir, run_id, version_label)
    status, reg = _api(f"{web_url}/api/ingest/runs", "POST", payload, key)
    if status != 200:
        print(f"register failed ({status}): {reg.get('error', reg)}", file=sys.stderr)
        return 1
    for name, target in reg["uploads"].items():
        body, ct = _artifact_bytes(out_dir, name, payload["families"])
        _put_blob(target, body, ct, key)
        print(f"uploaded {name} ({len(body)} bytes)")
    status, fin = _api(f"{web_url}/api/ingest/runs/{run_id}/finalize", "POST", None, key)
    if status != 200:
        print(f"finalize failed ({status}): {fin.get('error', fin)}", file=sys.stderr)
        return 1
    print(f"pushed run {run_id} ({version_label}) to {web_url}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Push a finished run to BreakPoint Web.")
    ap.add_argument("--target", default="go1")
    ap.add_argument("--version", default=None, help="policy version label (default: <target>_policy)")
    ap.add_argument("--run-id", default=None, help="reuse to resume a partial push (default: new uuid)")
    ap.add_argument("--bp-out", default=os.environ.get("BP_OUT", "out"))
    a = ap.parse_args()

    web_url = os.environ.get("BP_WEB_URL", "").rstrip("/")
    key = os.environ.get("BP_API_KEY", "")
    if not web_url or not key:
        print("set BP_WEB_URL and BP_API_KEY", file=sys.stderr)
        return 2

    out_dir = os.path.join(a.bp_out, a.target)
    if not os.path.exists(os.path.join(out_dir, "families.json")):
        print(f"no families.json in {out_dir}; run scripts/run_pipeline.py --target {a.target} first",
              file=sys.stderr)
        return 2

    version = a.version or f"{a.target}_policy"
    run_id = a.run_id or str(uuid.uuid4())
    return push(out_dir, web_url, key, version, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
