"""Summarize logs/chain_policy.json and logs/chain_curobo.json (+ the earlier policy-only / expert evals)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "logs"


def stats(tag):
    R = json.loads((L / f"{tag}.json").read_text())
    n = len(R)
    steps = lambda r, s: [e for e in r.get("log", []) if e["step"] == s]  # noqa: E731
    path = [r.get("grasp_path") for r in R]
    out = dict(
        n=n,
        success=sum(r["success"] for r in R),
        grasp_verified=sum(p is not None for p in path),
        grasp_path={k: path.count(k) for k in ("policy", "policy_retry", "curobo_fallback", "curobo", None) if path.count(k)},
        success_without_fallback=sum(r["success"] and r.get("grasp_path") in ("policy", "policy_retry") for r in R),
        transport_ok=sum(any(e["ok"] for e in steps(r, "transport") + steps(r, "replan")) for r in R),
        placed=sum(r.get("placed", False) for r in R),
        static_wall=sum(r.get("wall", False) for r in R),
        dyn_box_events=sum(any(e["brief"].startswith("box appeared") for e in steps(r, "obstacle_event")) for r in R),
        replans=sum(bool(steps(r, "replan")) for r in R),
        replans_ok=sum(any(e["ok"] for e in steps(r, "replan")) for r in R),
        replan_ms=[e["replan_ms"] for r in R for e in steps(r, "replan") if "replan_ms" in e],
        replan_clear_mm=[e["dyn_clear_replanned_mm"] for r in R for e in steps(r, "replan") if "dyn_clear_replanned_mm" in e],
        episodes_with_robot_obstacle_contact=sum((r.get("robot_obstacle_contacts") or 0) > 0 for r in R),
        episodes_with_brick_obstacle_contact=sum(any(k.startswith("cube|") for k in r.get("obstacle_contacts", {})) for r in R),
        unchecked_recoveries=sum(e.get("unchecked", False) for r in R for e in steps(r, "recover")),
        relaxed_place_orientation=sum(any(e.get("rpy_weights") for e in steps(r, "transport") + steps(r, "replan")) for r in R),
        wall_s_mean=round(float(np.mean([r["wall_s"] for r in R if "wall_s" in r])), 1),
        sim_s_mean=round(float(np.mean([r["sim_s"] for r in R if "sim_s" in r])), 1),
        errors=sum("error" in r for r in R),
    )
    if out["replan_ms"]:
        out["replan_ms_median"] = float(np.median(out["replan_ms"]))
    return out


res = {t: stats(t) for t in ("chain_policy", "chain_curobo") if (L / f"{t}.json").exists()}
for t in ("sim_ft", "curobo_expert"):
    p = L / f"eval_closedloop_{t}.json"
    if p.exists():
        R = json.loads(p.read_text())
        res[f"no_obstacles_{t}"] = dict(n=len(R), success=sum(r["success"] for r in R))
(L / "chain_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
