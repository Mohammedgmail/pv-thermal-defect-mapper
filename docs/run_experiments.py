"""Reproduce the evaluation in the README: pose errors and refinement, and sun reflections.

    python docs/run_experiments.py            # about 40 min on two cores, writes docs/results/

Everything runs on the thermal digital twin: the June noon plan from pv-drone-inspection-planner
is flown (with 3 cm / 0.2 deg tracking error), rendered with 38 seeded defects, and the
logged poses are corrupted with one of several error models before mapping.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pvdefect.camera import project
from pvdefect.evaluate import Survey, score
from pvdefect.mapper import Mapper, Settings
from pvdefect.scene import Scene
from pvdefect.simulate import GNSS, RTK, PoseErrors, random_defects

ROOT = Path(__file__).resolve().parents[1]
NOON = ROOT / "examples" / "june-noon" / "mission.json"
MORNING = ROOT / "examples" / "june-morning" / "mission.json"

ERRORS = {
    "none": PoseErrors(0, 0, 0, 0, 0, 0),
    "RTK": RTK,
    "attitude 1 deg": PoseErrors(
        yaw_bias_deg=1.0, yaw_jitter_deg=0.5, pitch_bias_deg=1.0, pitch_jitter_deg=0.5
    ),
    "GNSS 1.5 m": GNSS,
}
SEEDS = (0, 1, 2)
SURVEY: Survey | None = None  # shared with the worker processes (fork)


def speeds(scene: Scene) -> dict[int, float]:
    return {ln["index"]: ln["speed_mps"] for ln in scene.extras["mission"]["lines"]}


def _map(job):
    name, seed, refine = job
    t0 = time.time()
    findings, s = SURVEY.map(ERRORS[name], seed=seed, refine=refine)
    report = [r.as_dict() for r in getattr(SURVEY, "refine_report", [])] if refine else []
    applied = sum(r["applied"] for r in report)
    return {
        "errors": name,
        "seed": seed,
        "refine": refine,
        **s.as_dict(),
        "passes_refined": applied,
        "passes": len(report) or None,
        "seconds": round(time.time() - t0, 1),
    }


def pose_errors(out: Path, workers: int) -> None:
    global SURVEY
    scene = Scene.from_mission(NOON)
    defects = random_defects(scene, np.random.default_rng(7))
    t0 = time.time()
    SURVEY = Survey(scene, defects, seed=7, speeds=speeds(scene))
    print(f"rendered {len(SURVEY.index)} frames in {time.time() - t0:.0f} s", flush=True)
    jobs = [("none", 0, False)]
    jobs += [(n, s, r) for n in list(ERRORS)[1:] for s in SEEDS for r in (False, True)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_map, jobs):
            print(json.dumps(res), flush=True)
            rows.append(res)
    (out / "pose_errors.json").write_text(
        json.dumps({"defects": len(defects), "frames": len(SURVEY.index), "runs": rows}, indent=1)
    )


def min_glare(scene: Scene) -> float:
    """Smallest angle between the sun and a line of sight mirrored in the glass, over the parts
    of the planned rows that each image shows (deg)."""
    mapper = Mapper(scene)
    cam = scene.camera
    out = 180.0
    for p in scene.poses:
        sun = mapper._sun(p.time)
        for ri in mapper.line_rows[p.line]:
            r = scene.rows[ri]
            S, T = np.meshgrid(np.linspace(-r.length / 2, r.length / 2, 400),
                               np.linspace(-r.width / 2, r.width / 2, 9))  # fmt: skip
            P = r.point(S, T).reshape(-1, 3)
            u, v, depth = project(P, p, cam)
            ok = (
                (depth > 0)
                & (u >= 0)
                & (v >= 0)
                & (u <= cam.width_px - 1)
                & (v <= cam.height_px - 1)
            )
            if not ok.any():
                continue
            d = P[ok] - p.position
            d /= np.linalg.norm(d, axis=1)[:, None]
            m = d - 2 * (d @ r.normal)[:, None] * r.normal
            out = min(out, float(np.degrees(np.arccos(np.clip(m @ sun, -1, 1))).min()))
    return out


def reflections(out: Path) -> None:
    """The morning plan (steep view, two rows per pass) flown at 13:00 instead of 09:32, and the
    noon plan for comparison. Poses are exact here, to isolate the effect of reflections."""
    noon = Scene.from_mission(NOON)
    defects = random_defects(noon, np.random.default_rng(7))  # same plant, same defects
    variants = {
        "single image": Settings(reflection_guard_deg=0.0, min_views=1, min_view_fraction=0.0),
        "multi-view fusion": Settings(reflection_guard_deg=0.0),
        "glare guard": Settings(min_views=1, min_view_fraction=0.0),
        "fusion + glare guard": Settings(),
    }
    defect_ids = {d.module_id for d in defects}
    rows = []
    for plan, scene in (
        ("morning plan at 13:00", Scene.from_mission(MORNING).flown_at("13:00")),
        ("noon plan at 13:00", noon),
    ):
        t0 = time.time()
        survey = Survey(scene, defects, seed=7, speeds=speeds(scene))
        print(
            f"{plan}: rendered {len(survey.index)} frames in {time.time() - t0:.0f} s", flush=True
        )
        glare = round(min_glare(scene), 1)
        all_frames = survey.frames(survey.true)
        for overlap, frames in (("all images", all_frames), ("every 2nd image", all_frames[::2])):
            for name, st in variants.items():
                findings, views = Mapper(scene, st).run(frames)
                sc = score(findings, defects)
                rows.append(
                    {
                        "plan": plan,
                        "min_glare_deg": glare,
                        "images": overlap,
                        "variant": name,
                        **sc.as_dict(),
                        "hot_views_on_clean_modules": sum(
                            1 for v in views if v.kind == "cell" and v.module_id not in defect_ids
                        ),
                        "hot_spots_dropped_as_reflection": int(sum(v.reflections for v in views)),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    (out / "reflections.json").write_text(
        json.dumps({"defects": len(defects), "runs": rows}, indent=1)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results")
    ap.add_argument("--only", choices=["pose", "reflections"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "reflections"):
        reflections(a.out)
    if a.only in (None, "pose"):
        pose_errors(a.out, a.workers)


if __name__ == "__main__":
    main()
