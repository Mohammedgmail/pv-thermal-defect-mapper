"""Command line.

pvdefect simulate examples/june-noon/mission.json --out survey --lines 3,4   # render a survey
pvdefect map survey --mission examples/june-noon/mission.json --out results  # find and map defects
pvdefect map survey --mission examples/june-noon/mission.json --refine       # ... fixing GNSS poses
pvdefect raptor path/to/InfraredSolarModules                                 # real-data check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .mapper import Mapper, Settings, load_survey
from .scene import Scene

ERRORS = ("none", "rtk", "gnss", "attitude")


def _errors(name: str):
    from .simulate import GNSS, RTK, PoseErrors

    return {
        "none": PoseErrors(0, 0, 0, 0, 0, 0),
        "rtk": RTK,
        "gnss": GNSS,
        "attitude": PoseErrors(
            yaw_bias_deg=1.0, yaw_jitter_deg=0.5, pitch_bias_deg=1.0, pitch_jitter_deg=0.5
        ),
    }[name]


def _progress(n: int, total: int) -> None:
    if n == total or n % 25 == 0:
        print(f"  {n}/{total} frames", file=sys.stderr, flush=True)


def cmd_simulate(a: argparse.Namespace) -> None:
    from .simulate import (
        ThermalTwin,
        Weather,
        flown_poses,
        logged_poses,
        random_defects,
        write_survey,
    )

    scene = Scene.from_mission(a.mission)
    if a.start:
        scene = scene.flown_at(a.start)
    defects = random_defects(scene, np.random.default_rng(a.seed))
    rng = np.random.default_rng(a.seed)
    true = flown_poses(scene, rng)
    logged = logged_poses(true, _errors(a.errors), np.random.default_rng(10_000 + a.seed))
    lines = None if not a.lines else {int(x) for x in a.lines.split(",")}
    idx = [i for i, p in enumerate(true) if lines is None or p.line in lines]
    speeds = {ln["index"]: ln["speed_mps"] for ln in scene.extras["mission"]["lines"]}
    twin = ThermalTwin(scene, defects, Weather(a.air, a.wind), seed=a.seed)
    print(f"rendering {len(idx)} frames ({a.errors} pose errors) ...", file=sys.stderr)
    t0 = time.time()
    write_survey(a.out, scene, twin, logged, true, speeds, idx, _progress)
    secs = time.time() - t0
    print(f"wrote {a.out}/frames.json, {a.out}/truth.json and {len(idx)} frames in {secs:.0f} s")


def cmd_map(a: argparse.Namespace) -> None:
    from .evaluate import score
    from .report import write_outputs
    from .simulate import Defect

    scene = Scene.from_mission(a.mission)
    frames = load_survey(a.survey)
    refine_report = None
    if a.refine:
        from .refine import refine_poses

        poses, rep = refine_poses(scene, frames)
        frames = [(img, p) for (img, _), p in zip(frames, poses, strict=True)]
        refine_report = [r.as_dict() for r in rep]
    settings = Settings(hot_k=a.hot_k, reflection_guard_deg=a.guard, min_views=a.min_views)
    findings, views = Mapper(scene, settings).run(frames, relative=a.relative)
    truth_path = Path(a.survey) / "truth.json"
    sc = None
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())
        seen = {v.module_id for v in views}  # score only what the frames cover
        defects = [
            Defect(
                d["module_id"],
                d["kind"],
                d["delta_t"],
                [tuple(c) for c in d["cells"]],
                d["substring"],
            )
            for d in truth["defects"]
            if d["module_id"] in seen
        ]
        sc = score(findings, defects).as_dict()
    out = Path(a.out)
    paths = write_outputs(out, scene, findings, views, frames, refine_report, sc)
    print((out / "report.md").read_text().split("## Findings")[0].strip())
    print("\nwrote " + ", ".join(str(p) for p in paths))


def cmd_raptor(a: argparse.Namespace) -> None:
    from .realdata import evaluate

    res = evaluate(a.dataset, per_class=a.per_class)
    for cls, r in res["default"].items():
        rates = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r.items()) if k != "n")
        print(f"{cls:15s} n={r['n']:5d}  {rates}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="pvdefect", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "simulate", help="render a thermal survey of a pvmission plan (digital twin)"
    )
    s.add_argument("mission", help="mission.json from pv-drone-inspection-planner")
    s.add_argument("--out", default="survey")
    s.add_argument("--lines", help="only these flight lines, e.g. 3,4 (default: all)")
    s.add_argument("--start", help="fly the plan at another local time HH:MM")
    s.add_argument(
        "--errors", choices=ERRORS, default="rtk", help="logged-pose errors (default rtk)"
    )
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--air", type=float, default=21.0, help="air temperature, deg C")
    s.add_argument("--wind", type=float, default=3.0, help="wind speed, m/s")
    s.set_defaults(func=cmd_simulate)

    m = sub.add_parser("map", help="find hot spots and map them to modules and cells")
    m.add_argument("survey", help="folder with frames.json and the images")
    m.add_argument("--mission", required=True, help="the mission.json the survey was flown with")
    m.add_argument("--out", default="results")
    m.add_argument(
        "--refine", action="store_true", help="correct the logged poses from the image first"
    )
    m.add_argument(
        "--relative", action="store_true", help="images without temperatures (thresholds in sigma)"
    )
    m.add_argument("--hot-k", type=float, default=Settings.hot_k, help="hot spot threshold, K")
    m.add_argument(
        "--guard", type=float, default=Settings.reflection_guard_deg, help="reflection guard, deg"
    )
    m.add_argument("--min-views", type=int, default=Settings.min_views)
    m.set_defaults(func=cmd_map)

    r = sub.add_parser(
        "raptor", help="check the module analyser on Raptor Maps InfraredSolarModules"
    )
    r.add_argument("dataset", help="unzipped InfraredSolarModules folder")
    r.add_argument("--per-class", type=int, help="random sample of at most N images per class")
    r.add_argument("--out", help="write the results as JSON")
    r.set_defaults(func=cmd_raptor)

    a = ap.parse_args(argv)
    a.func(a)
