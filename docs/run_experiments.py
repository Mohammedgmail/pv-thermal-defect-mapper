"""Reproduce every result in the README and the paper (docs/results/*.json).

    python docs/run_experiments.py --fresh         # everything from scratch
    python docs/run_experiments.py main stress     # only some experiments

An interrupted run resumes where it stopped (finished jobs are kept in docs/results/*.jsonl).

Experiments
-----------
main         noon plan, 10 flights x {RTK, attitude 1 deg, GNSS 1.5 m} x {logged, PnP, ours}
ablation     refinement steps switched on one by one (attitude and GNSS, 3 flights)
stress       where it breaks: GNSS and attitude bias sweeps, frame contrast, sensor noise,
             speed (motion blur), detection limit, soiling, passes without visible row ends
sites        three more plants planned by pv-drone-inspection-planner (Seville, Gujarat with
             landscape modules, Queensland facing north)
reflections  the morning plan flown at 13:00, single image vs fusion vs glare guard
reinspection close the loop: plan a re-flight for what the survey left uncertain

Everything runs on the thermal digital twin: frames are rendered from the flown poses (plan
plus 3 cm / 0.2 deg tracking error) and the mapper gets logged poses with errors added.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from pvdefect.baseline import pnp_refine
from pvdefect.camera import project
from pvdefect.errormodel import plane_point
from pvdefect.evaluate import Survey, score
from pvdefect.mapper import Mapper, Settings
from pvdefect.refine import ALL_STEPS, refine_poses
from pvdefect.scene import Module, Scene
from pvdefect.simulate import (
    GNSS,
    RTK,
    Defect,
    ModuleThermal,
    Optics,
    PoseErrors,
    ThermalTwin,
    fixed_bias_poses,
    flown_poses,
    logged_poses,
    random_defects,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
NOON = ROOT / "examples" / "june-noon" / "mission.json"
MORNING = ROOT / "examples" / "june-morning" / "mission.json"
SITES = {
    n: ROOT / "examples" / "sites" / n / "mission.json"
    for n in ("seville", "gujarat", "queensland")
}
ERRORS = {
    "none": PoseErrors(0, 0, 0, 0, 0, 0),
    "RTK": RTK,
    "attitude 1 deg": PoseErrors(
        yaw_bias_deg=1.0, yaw_jitter_deg=0.5, pitch_bias_deg=1.0, pitch_jitter_deg=0.5
    ),
    "GNSS 1.5 m": GNSS,
}
ABLATION = {
    "t": ("t",),
    "t+scale": ("t", "scale"),
    "t+scale+s": ("t", "scale", "s"),
    "t+scale+s+anchors": ("t", "scale", "s", "anchors"),
    "full": ALL_STEPS,  # + texture links between frames
}
STATE: dict = {}  # shared with forked worker processes
CACHE = Path(os.environ.get("PVDEFECT_CACHE", Path(tempfile.gettempdir()) / "pvdefect-cache"))
RENDER_WORKERS = [2]


def survey(scene: Scene, defects) -> Survey:
    """The whole flight rendered (in parallel, cached between runs: the noon survey is used
    by four experiments)."""
    return Survey(
        scene, defects, seed=7, speeds=speeds(scene), workers=RENDER_WORKERS[0], cache=CACHE
    )


# ------------------------------------------------------------------------------- helpers
def speeds(scene: Scene) -> dict[int, float]:
    return {ln["index"]: ln["speed_mps"] for ln in scene.extras["mission"]["lines"]}


def line_rows(scene: Scene) -> dict[int, list[int]]:
    return {ln["index"]: [r - 1 for r in ln["rows"]] for ln in scene.extras["mission"]["lines"]}


def pose_errors_m(scene: Scene, est, true) -> np.ndarray:
    """Per frame and row: distance on the row plane between where the estimated and the true
    pose put the pixel that truly shows the row's centre line below the camera."""
    rows_of = line_rows(scene)
    out = []
    for p, q in zip(est, true, strict=True):
        for ri in rows_of[q.line]:
            row = scene.rows[ri]
            s = (q.position - row.center) @ row.along
            P = row.point(np.array([s]), np.array([0.0]))
            u, v, _ = project(P, q, scene.camera)
            Q = plane_point(p.for_row(ri), scene, row, float(u[0]), float(v[0]))
            out.append(float(np.linalg.norm(Q - P[0])))
    return np.array(out)


def correct(method: str, scene: Scene, frames):
    """Poses after the named correction method (and its pass reports, if any)."""
    poses = [p for _, p in frames]
    if method == "logged":
        return poses, None
    if method == "pnp":
        return pnp_refine(scene, frames), None
    steps = ABLATION[method.removeprefix("ours:")] if method.startswith("ours:") else ALL_STEPS
    poses, rep = refine_poses(scene, frames, steps=steps)
    return poses, [r.as_dict() for r in rep]


def evaluate(scene, images, true, logged, method, truth, settings=Settings(), nuisance=()):
    frames = list(zip(images, logged, strict=True))
    t0 = time.time()
    poses, rep = correct(method, scene, frames)
    t_corr = time.time() - t0
    findings, views = Mapper(scene, settings).run(list(zip(images, poses, strict=True)))
    seen = {v.module_id for v in views}
    s = score(findings, [d for d in truth if d.module_id in seen]).as_dict()
    err = pose_errors_m(scene, poses, true)
    out = {
        "method": method,
        **s,
        "pose_err_median_m": round(float(np.median(err)), 4),
        "pose_err_p90_m": round(float(np.percentile(err, 90)), 4),
        "frames_off_30cm": round(float((err > 0.3).mean()), 4),
        "correction_s_per_frame": round(t_corr / max(1, len(images)), 3),
    }
    if rep is not None:
        out["passes_refined"] = sum(r["applied"] for r in rep)
        out["passes"] = len(rep)
    if nuisance:
        ids = {d.module_id for d in nuisance}
        out["fp_on_soiled"] = sum(1 for f in findings if f.module_id in ids)
    return out


def write(name: str, payload: dict) -> None:
    """Final results: the rows streamed to <name>.jsonl (all chunks so far) plus ``payload``."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stream = RESULTS / f"{name}.jsonl"
    rows = [json.loads(x) for x in stream.read_text().splitlines()] if stream.exists() else []
    (RESULTS / f"{name}.json").write_text(json.dumps({**payload, "runs": rows}, indent=1))
    print(f"wrote {RESULTS / name}.json ({len(rows)} rows)", flush=True)


def log(row: dict, name: str | None = None) -> None:
    print(json.dumps(row), flush=True)
    if name:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def pool(fn, jobs, workers, name):
    """Run jobs in forked worker processes (they share STATE); stream every result row to
    <name>.jsonl as soon as its job finishes, so an interrupted run loses only the jobs that
    were still running."""
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for fut in as_completed([ex.submit(fn, j) for j in jobs]):
            res = fut.result()
            for r in res if isinstance(res, list) else [res]:
                log(r, name)
                yield r


def _done(name: str, keys: tuple[str, ...]) -> set:
    """Jobs already streamed to <name>.jsonl (to resume an interrupted run)."""
    stream = RESULTS / f"{name}.jsonl"
    if not stream.exists():
        return set()
    return {tuple(json.loads(x)[k] for k in keys) for x in stream.read_text().splitlines()}


def _todo(name: str, jobs: list, keys: tuple[str, ...]) -> list:
    """The jobs not yet in <name>.jsonl; each job is a tuple of the values of ``keys``."""
    done = _done(name, keys)
    todo = [j for j in jobs if tuple(j) not in done]
    if len(todo) < len(jobs):
        print(f"{name}: {len(jobs) - len(todo)} job(s) done before, {len(todo)} to run", flush=True)
    return todo


def fresh(name: str) -> None:
    (RESULTS / f"{name}.jsonl").unlink(missing_ok=True)


# ------------------------------------------------------------------------------- main
def _main_job(job):
    errors, seed, method = job
    sv = STATE["survey"]
    logged = sv.logged(ERRORS[errors], seed)
    images = [sv.images[k] for k in range(len(sv.index))]
    r = evaluate(sv.scene, images, sv.true, logged, method, sv.defects)
    return {"errors": errors, "seed": seed, **r}


def run_main(workers: int, seeds=range(10)) -> None:
    """``seeds``: the flights to run (resumes: jobs already in main.jsonl are skipped)."""
    jobs = [("none", 0, "logged")] if 0 in seeds else []
    jobs += [
        (e, s, m)
        for e in ("RTK", "attitude 1 deg", "GNSS 1.5 m")
        for s in seeds
        for m in ("logged", "pnp", "ours")
    ]
    todo = _todo("main", jobs, ("errors", "seed", "method"))
    scene = Scene.from_mission(NOON)
    defects = random_defects(scene, np.random.default_rng(7))
    if todo:
        t0 = time.time()
        STATE["survey"] = survey(scene, defects)
        print(f"main: rendered {len(scene.poses)} frames in {time.time() - t0:.0f} s", flush=True)
        list(pool(_main_job, todo, workers, "main"))
    write("main", {"defects": len(defects), "frames": len(scene.poses)})


def run_ablation(workers: int, seeds: int = 3) -> None:
    jobs = [
        (e, s, f"ours:{a}")
        for e in ("attitude 1 deg", "GNSS 1.5 m")
        for s in range(seeds)
        for a in ABLATION
    ]
    todo = _todo("ablation", jobs, ("errors", "seed", "method"))
    if todo:
        scene = Scene.from_mission(NOON)
        defects = random_defects(scene, np.random.default_rng(7))
        STATE["survey"] = survey(scene, defects)
        list(pool(_main_job, todo, workers, "ablation"))
    write("ablation", {})


# ------------------------------------------------------------------------------- stress
def _subset(scene: Scene, defects, n_lines: int = 6) -> list[int]:
    """The flight lines whose rows hold the most defects."""
    rows_of = line_rows(scene)
    per_row = {}
    for d in defects:
        r = int(d.module_id[1:3]) - 1
        per_row[r] = per_row.get(r, 0) + 1
    lines = sorted(rows_of, key=lambda ln: -sum(per_row.get(r, 0) for r in rows_of[ln]))
    return sorted(lines[:n_lines])


NON_UNIFORMITY = {  # cell scatter, junction-box patch, edge change (K)
    "clean": ModuleThermal(0.25, 0.0, 0.0),
    "moderate": ModuleThermal(0.5, 1.0, -0.5),
    "rough": ModuleThermal(1.0, 2.0, -1.0),
}


def _render(
    scene, defects, lines, optics=Optics(), speed_factor=1.0, nuisance=(), thermal=ModuleThermal()
):
    true = flown_poses(scene, np.random.default_rng(7))
    idx = [i for i, p in enumerate(true) if p.line in lines]
    twin = ThermalTwin(
        scene, list(defects) + list(nuisance), optics=optics, seed=7, thermal=thermal
    )
    sp = speeds(scene)
    images = [twin.render(true[i], speed_mps=speed_factor * sp[true[i].line]) for i in idx]
    return images, [true[i] for i in idx]


def _stress_job(job):
    kind, value = job
    scene, defects, lines = STATE["scene"], STATE["defects"], STATE["lines"]
    rows = []
    base = {"experiment": kind, "value": value}
    if kind in ("gnss_bias", "attitude_bias", "no_row_ends"):
        images, true = STATE["images"], STATE["true"]
        for trial in range(2):
            rng = np.random.default_rng(1000 + trial)
            if kind == "gnss_bias":
                logged = fixed_bias_poses(true, rng, horizontal_m=value, vertical_m=value)
            elif kind == "attitude_bias":
                logged = fixed_bias_poses(true, rng, yaw_deg=value, pitch_deg=value)
            else:  # drop the frames that see a row end: no absolute along-row anchor
                logged = fixed_bias_poses(true, rng, horizontal_m=value, vertical_m=value)
            ims, tr, lg = images, true, logged
            if kind == "no_row_ends":
                keep = _middle_frames(true)
                ims, tr, lg = ([x[k] for k in keep] for x in (images, true, logged))
            for method in ("logged", "pnp", "ours"):
                r = evaluate(scene, ims, tr, lg, method, defects)
                rows.append({**base, "trial": trial, **r})
        return rows
    optics, speed, nuisance, truth, settings = Optics(), 1.0, (), defects, [Settings()]
    thermal = ModuleThermal()
    if kind == "frame_contrast":
        optics = replace(optics, frame_k=value)
    elif kind == "netd":
        optics = replace(optics, netd_k=value)
    elif kind == "speed":
        speed = value
    elif kind == "detection_limit":
        truth = STATE["weak_defects"]
        settings = [Settings(hot_k=k) for k in (1.0, 1.5, 2.0, 3.0)]
    elif kind == "soiling":
        nuisance = STATE["soiling"]
    elif kind == "non_uniformity":
        thermal = NON_UNIFORMITY[value]
        truth = STATE["weak_defects"]
        settings = [Settings(hot_k=k) for k in (2.0, 3.0, 4.0)]
    images, true = _render(scene, truth, lines, optics, speed, nuisance, thermal)
    for st in settings:
        r = evaluate(scene, images, true, true, "logged", truth, st, nuisance)
        extra = {"hot_k": st.hot_k}
        if kind in ("detection_limit", "non_uniformity"):
            extra["recall_by_dt"] = _recall_by_dt(scene, images, true, truth, st)
        rows.append({**base, "poses": "exact", **extra, **r})
    if kind in ("frame_contrast", "netd", "speed"):
        for trial in range(2):
            logged = logged_poses(true, GNSS, np.random.default_rng(10_000 + trial))
            for method in ("logged", "ours"):
                r = evaluate(scene, images, true, logged, method, truth)
                rows.append({**base, "poses": "GNSS 1.5 m", "trial": trial, **r})
    return rows


def _middle_frames(true) -> list[int]:
    by_line: dict[int, list[int]] = {}
    for k, p in enumerate(true):
        by_line.setdefault(p.line, []).append(k)
    keep = []
    for ks in by_line.values():
        keep += ks[4:-4]
    return sorted(keep)


def _recall_by_dt(scene, images, true, truth, st) -> dict:
    findings, views = Mapper(scene, st).run(list(zip(images, true, strict=True)))
    seen = {v.module_id for v in views}
    found = {f.module_id for f in findings}
    out = {}
    for d in truth:
        if d.module_id not in seen:
            continue
        k = f"{d.delta_t:g}"
        hit, n = out.get(k, (0, 0))
        out[k] = (hit + (d.module_id in found), n + 1)
    return {k: round(h / n, 3) for k, (h, n) in sorted(out.items(), key=lambda kv: float(kv[0]))}


def _weak_defects(scene, lines, rng) -> list[Defect]:
    """Single hot cells of 1 to 6 K on the subset rows, well apart."""
    rows_of = line_rows(scene)
    rows = sorted({r for ln in lines for r in rows_of[ln]})
    out, taken = [], set()
    levels = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    room = sum(scene.rows[r].modules_along for r in rows) // 4  # keep defects apart
    for i in range(min(12 * len(levels), room // len(levels) * len(levels))):
        while True:
            row = scene.rows[rows[rng.integers(len(rows))]]
            c = int(rng.integers(3, row.modules_along - 1))
            lvl = int(rng.integers(1, row.modules_across + 1))
            m = Module(row, c, lvl)
            near = {(row.index, c + d) for d in (-1, 0, 1)}
            if not near & taken:
                taken |= near
                break
        n_s, n_t = m.grid
        cell = (int(rng.integers(n_t)), int(rng.integers(n_s)))
        out.append(Defect(m.id, "hot_cell", levels[i % len(levels)], [cell]))
    return out


def _soiling(scene, lines, defects, rng, n=30) -> list[Defect]:
    rows_of = line_rows(scene)
    rows = sorted({r for ln in lines for r in rows_of[ln]})
    bad = {d.module_id for d in defects}
    out = []
    while len(out) < n:
        row = scene.rows[rows[rng.integers(len(rows))]]
        m = Module(row, int(rng.integers(2, row.modules_along)), 1)
        if m.id not in bad and all(d.module_id != m.id for d in out):
            out.append(Defect(m.id, "soiling", float(rng.uniform(3, 6))))
    return out


STRESS_POSE = ("gnss_bias", "attitude_bias", "no_row_ends")  # chunk 1 (uses one rendering)


def run_stress(workers: int, chunk: str = "all") -> None:
    """``chunk``: "pose" (bias sweeps), "image" (rendering conditions) or "all"."""
    scene = Scene.from_mission(NOON)
    defects = random_defects(scene, np.random.default_rng(7))
    lines = _subset(scene, defects)
    STATE.update(scene=scene, defects=defects, lines=lines)
    STATE["weak_defects"] = _weak_defects(scene, lines, np.random.default_rng(3))
    STATE["soiling"] = _soiling(scene, lines, defects, np.random.default_rng(4))
    jobs = []
    if chunk in ("pose", "all"):
        jobs += [("gnss_bias", v) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)]
        jobs += [("attitude_bias", v) for v in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
        jobs += [("no_row_ends", v) for v in (0.5, 1.5)]
    if chunk in ("image", "all"):
        jobs += [("frame_contrast", v) for v in (-6.0, -3.0, -1.5, -0.5)]
        jobs += [("netd", v) for v in (0.05, 0.1, 0.2, 0.4)]
        jobs += [("speed", v) for v in (1.0, 2.0, 4.0)]
        jobs += [("detection_limit", 0), ("soiling", 0)]
        jobs += [("non_uniformity", v) for v in NON_UNIFORMITY]
    todo = _todo("stress", jobs, ("experiment", "value"))
    if any(kind in STRESS_POSE for kind, _ in todo):  # these share one rendering
        t0 = time.time()
        STATE["images"], STATE["true"] = _render(scene, defects, lines)
        print(
            f"stress: {len(STATE['images'])} frames on lines {lines} in {time.time() - t0:.0f} s",
            flush=True,
        )
    list(pool(_stress_job, todo, workers, "stress"))
    n_frames = sum(1 for p in flown_poses(scene, np.random.default_rng(7)) if p.line in lines)
    write("stress", {"lines": lines, "frames": n_frames})


# ------------------------------------------------------------------------------- sites
def _site_job(job):
    errors, seed, method = job
    sv = STATE["survey"]
    logged = sv.logged(ERRORS[errors], seed)
    images = [sv.images[k] for k in range(len(sv.index))]
    r = evaluate(sv.scene, images, sv.true, logged, method, sv.defects)
    return {**STATE["info"], "errors": errors, "seed": seed, **r}


def run_sites(workers: int, seeds: int = 3) -> None:
    for name, path in SITES.items():
        jobs = [(name, "none", 0, "logged")]
        jobs += [
            (name, e, s, mt)
            for e in ("RTK", "GNSS 1.5 m")
            for s in range(seeds)
            for mt in ("logged", "pnp", "ours")
        ]
        jobs = [j[1:] for j in _todo("sites", jobs, ("site", "errors", "seed", "method"))]
        if not jobs:
            continue
        scene = Scene.from_mission(path)
        defects = random_defects(scene, np.random.default_rng(7))
        t0 = time.time()
        STATE["survey"] = survey(scene, defects)
        m = scene.extras["mission"]
        print(f"sites: {name} rendered {len(scene.poses)} frames in {time.time() - t0:.0f} s")
        info = {
            "site": name,
            "plant": m["plant"]["name"],
            "start": m["summary"]["start"],
            "rows_per_line": m["summary"]["rows_per_line"],
            "frames": len(scene.poses),
            "modules": scene.n_modules,
            "portrait": Module(scene.rows[0], 1, 1).portrait,
            "azimuth_deg": scene.rows[0].azimuth,
            "min_glare_deg": m["summary"]["min_glare_deg"],
        }
        STATE["info"] = info
        list(pool(_site_job, jobs, workers, "sites"))
    write("sites", {})


# ------------------------------------------------------------------------------- reflections
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
            S, T = np.meshgrid(
                np.linspace(-r.length / 2, r.length / 2, 400),
                np.linspace(-r.width / 2, r.width / 2, 9),
            )
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


def run_reflections(workers: int) -> None:
    noon = Scene.from_mission(NOON)
    defects = random_defects(noon, np.random.default_rng(7))  # same plant, same defects
    variants = {
        "single image": Settings(reflection_guard_deg=0.0, min_views=1, min_view_fraction=0.0),
        "multi-view fusion": Settings(reflection_guard_deg=0.0),
        "glare guard": Settings(min_views=1, min_view_fraction=0.0),
        "fusion + glare guard": Settings(),
    }
    ids = {d.module_id for d in defects}
    rows = []
    fresh("reflections")
    for plan, scene in (
        ("morning plan at 13:00", Scene.from_mission(MORNING).flown_at("13:00")),
        ("noon plan at 13:00", noon),
    ):
        sv = survey(scene, defects)
        glare = round(min_glare(scene), 1)
        frames = sv.frames(sv.true)
        for overlap, fr in (("all images", frames), ("every 2nd image", frames[::2])):
            for name, st in variants.items():
                findings, views = Mapper(scene, st).run(fr)
                sc = score(findings, defects)
                rows.append(
                    {
                        "plan": plan,
                        "min_glare_deg": glare,
                        "images": overlap,
                        "variant": name,
                        **sc.as_dict(),
                        "hot_views_on_clean_modules": sum(
                            1 for v in views if v.kind == "cell" and v.module_id not in ids
                        ),
                        "hot_spots_dropped_as_reflection": int(sum(v.reflections for v in views)),
                    }
                )
                log(rows[-1], "reflections")
    write("reflections", {"defects": len(defects)})


# ------------------------------------------------------------------------------- re-inspection
def run_reinspection(workers: int) -> None:
    """Close the loop on the morning plan flown at 13:00 with every second image: the survey
    misses a hot cell under the sun's reflection; a re-inspection flight planned from the
    survey's own uncertainty finds it. Also counts candidates for the clean noon survey."""
    import pandas as pd

    from pvdefect import reinspect as RI

    noon = Scene.from_mission(NOON)
    defects = random_defects(noon, np.random.default_rng(7))
    out_dir = RESULTS / "reinspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fresh("reinspection")
    for plan_name, scene, step in (
        ("noon plan, all images", noon, 1),
        ("morning plan at 13:00, all images", Scene.from_mission(MORNING).flown_at("13:00"), 1),
        (
            "morning plan at 13:00, every 2nd image",
            Scene.from_mission(MORNING).flown_at("13:00"),
            2,
        ),
    ):
        sv = survey(scene, defects)
        frames = sv.frames(sv.true)[::step]
        mapper = Mapper(scene)
        findings, views = mapper.run(frames)
        before = score(findings, defects)
        todo = RI.candidates(findings, views)
        hlat, hlon, _ = scene.extras["mission"]["home_lla"]
        home = np.array([*scene.frame.to_local(hlat, hlon), 0.0])
        start = sv.true[-1].time + pd.Timedelta(minutes=5)
        plan = RI.plan(scene, todo, start, home)
        row = {
            "plan": plan_name,
            "survey": before.as_dict(),
            "candidates": [c.__dict__ for c in todo],
            "viewpoints": [
                {
                    "module": v.module_id,
                    "reason": v.reason,
                    "enu": [round(float(x), 2) for x in v.position],
                    "yaw_deg": round(v.yaw_deg, 1),
                    "pitch_deg": round(v.pitch_deg, 1),
                    "view_deg": round(v.view_deg, 1),
                    "glare_deg": round(v.glare_deg, 1),
                    "gsd_cm": round(v.gsd_cm, 2),
                }
                for v in plan.viewpoints
            ],
            "unreachable": len(plan.unreachable),
            "reflight_minutes": round(plan.flight_s / 60, 2),
        }
        if plan.viewpoints:
            scene2 = RI.scene_with(scene, plan)
            planned = RI.poses(plan)
            rng = np.random.default_rng(5)  # the drone holds each viewpoint to SITL accuracy
            flown = [p.moved(rng.normal(0, 0.03, 3), rng.normal(0, 0.2)) for p in planned]
            twin = ThermalTwin(scene2, defects, seed=7)
            extra = []
            mapper2 = Mapper(scene2)
            for k, (pf, pl) in enumerate(zip(flown, planned, strict=True)):
                extra += mapper2.frame_views(100_000 + k, twin.render(pf), pl)
            after = score(mapper2.fuse(views + extra), defects)
            row["after"] = after.as_dict()
            slug = plan_name.split(",")[0].replace(" ", "_").replace(":", "")
            RI.write_waypoints(out_dir / f"{slug}_step{step}.waypoints", scene, plan, home)
        log(row, "reinspection")
        rows.append(row)
    write("reinspection", {})


ONE_SHOT = ("reflections", "reinspection")  # short; rerun as a whole, not resumed
EXPERIMENTS = {
    "main": run_main,
    "ablation": run_ablation,
    "stress": run_stress,
    "main:1": lambda w: run_main(w, range(0, 5)),
    "main:2": lambda w: run_main(w, range(5, 10)),
    "stress:pose": lambda w: run_stress(w, "pose"),
    "stress:image": lambda w: run_stress(w, "image"),
    "sites": run_sites,
    "reflections": run_reflections,
    "reinspection": run_reinspection,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("experiments", nargs="*", choices=list(EXPERIMENTS), default=[])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--fresh", action="store_true", help="discard earlier results instead of resuming"
    )
    a = ap.parse_args()
    RENDER_WORKERS[0] = a.workers
    default = ["main", "ablation", "stress", "sites", "reflections", "reinspection"]
    wiped: set[str] = set()
    for name in a.experiments or default:
        base = name.split(":")[0]
        if a.fresh and base not in wiped:  # stress:pose and stress:image share stress.jsonl
            fresh(base)
            (RESULTS / f"{base}.json").unlink(missing_ok=True)
            wiped.add(base)
        elif not a.fresh and name in ONE_SHOT and (RESULTS / f"{name}.json").exists():
            print(f"=== {name}: done before (use --fresh to run again)", flush=True)
            continue
        t0 = time.time()
        print(f"=== {name}", flush=True)
        EXPERIMENTS[name](a.workers)
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
