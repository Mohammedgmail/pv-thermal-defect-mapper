"""Draw the README figures and the example results (about 15 minutes).

python docs/run_experiments.py                       # first: docs/results/*.json
python docs/make_figures.py --raptor path/to/InfraredSolarModules
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from pvdefect.camera import project, rays  # noqa: E402
from pvdefect.mapper import Mapper, Settings, module_crop  # noqa: E402
from pvdefect.refine import refine_poses  # noqa: E402
from pvdefect.report import KIND_LABEL, best_views, write_outputs  # noqa: E402
from pvdefect.scene import Scene  # noqa: E402
from pvdefect.simulate import (  # noqa: E402
    GNSS,
    RTK,
    ThermalTwin,
    flown_poses,
    logged_poses,
    random_defects,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS, RESULTS = ROOT / "docs", ROOT / "docs" / "results"
NOON = ROOT / "examples" / "june-noon" / "mission.json"
MORNING = ROOT / "examples" / "june-morning" / "mission.json"
SURFACE, INK, INK_2, GRID, ROWS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d7d1"
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
KIND_COLOUR = {
    "hot_cell": BLUE,
    "hot_cells": AQUA,
    "substring": YELLOW,
    "offline": ORANGE,
    "string": PINK,
}
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def title(ax, text, size=11.5):
    ax.set_title(text, loc="left", color=INK, fontsize=size, fontweight="bold")


def speeds(scene):
    return {ln["index"]: ln["speed_mps"] for ln in scene.extras["mission"]["lines"]}


# ---------------------------------------------------------------------------------- hero
def example_survey(scene, defects):
    """Full noon survey: rendered from the flown poses, logged with RTK errors, refined."""
    true = flown_poses(scene, np.random.default_rng(7))
    twin = ThermalTwin(scene, defects, seed=7)
    sp = speeds(scene)
    imgs = [twin.render(p, speed_mps=sp[p.line]) for p in true]
    logged = logged_poses(true, RTK, np.random.default_rng(10_000))
    frames = list(zip(imgs, logged, strict=True))
    poses, report = refine_poses(scene, frames)
    frames = list(zip(imgs, poses, strict=True))
    findings, views = Mapper(scene, Settings()).run(frames)
    return frames, findings, views, [r.as_dict() for r in report]


def outline(ax, scene, module, pose, colour, lw=1.6, halo=True):
    u, v, _ = project(module.corners(), pose, scene.camera)
    poly = Polygon(np.c_[u, v], closed=True, fill=False, ec=colour, lw=lw)
    if halo:
        poly.set_path_effects([pe.withStroke(linewidth=lw + 2.2, foreground="white")])
    ax.add_patch(poly)


def hero(scene, frames, findings, views):
    found = {f.module_id: f for f in findings}
    # the frame that shows the most findings in full
    best_k, best_n = 0, -1
    for k, (img, pose) in enumerate(frames):
        if pose.line != 3:
            continue
        n = sum(module_crop(img, scene.module(m), pose, scene, 0.05) is not None for m in found)
        if n > best_n:
            best_k, best_n = k, n
    img, pose = frames[best_k]
    fig = plt.figure(figsize=(12.6, 6.3), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.08, 0.6, 0.8])
    lo, hi = np.percentile(img, [40, 99.9])  # stretch over the modules
    ax.imshow(img, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_axis_off()
    row = scene.rows[3]
    for c in range(1, row.modules_along + 1):  # module grid of the planned row
        for lvl in (1, 2):
            m = scene.module(f"R04-C{c:03d}-{lvl}")
            u, v, d = project(m.corners(), pose, scene.camera)
            if (d > 0).all() and u.max() > 0 and u.min() < 639:
                ax.add_patch(
                    Polygon(np.c_[u, v], closed=True, fill=False, ec="white", lw=0.35, alpha=0.6)
                )
    shown = []
    for mid, f in found.items():
        m = scene.module(mid)
        if module_crop(img, m, pose, scene, 0.05) is None:
            continue
        outline(ax, scene, m, pose, KIND_COLOUR[f.kind], lw=2.0)
        u, v, _ = project(m.corners(), pose, scene.camera)
        shown.append((float(u.mean()), float(v.min()), f))
    for i, (u, v, f) in enumerate(sorted(shown, key=lambda x: x[0])):
        extra = f" at cell {f.cells[0]}" if f.kind == "hot_cell" else ""
        ax.annotate(
            f"{f.module_id}\n{KIND_LABEL[f.kind]}{extra}, +{f.delta_t:.1f} K",
            (u, v),
            (u, 40 + 62 * (i % 2)),
            ha="center",
            fontsize=8,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": SURFACE,
                "ec": KIND_COLOUR[f.kind],
                "lw": 1.2,
            },
            arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.0},
        )
    ax.set_title(
        f"One of 500 thermal frames (row 4, {pose.time:%H:%M:%S}), module grid from the plan",
        loc="left",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
    )
    # plant map
    mx = fig.add_axes([0.64, 0.2, 0.35, 0.68])
    style(mx)
    mx.grid(False)
    for r in scene.rows:
        c = r.corners()
        mx.fill(c[:, 0], c[:, 1], color=ROWS, lw=0)
    for f in findings:
        c = scene.module(f.module_id).corners()
        mx.fill(c[:, 0], c[:, 1], color=KIND_COLOUR[f.kind], lw=0.6, ec=KIND_COLOUR[f.kind])
    # this frame's footprint on the ground
    cam = scene.camera
    uu = np.array([0, cam.width_px - 1, cam.width_px - 1, 0], float)
    vv = np.array([0, 0, cam.height_px - 1, cam.height_px - 1], float)
    d = rays(uu, vv, pose, cam)
    t = (1.0 - pose.position[2]) / d[:, 2]
    foot = pose.position + t[:, None] * d
    mx.add_patch(Polygon(foot[:, :2], closed=True, fill=False, ec=INK, lw=1.0, ls=(0, (3, 2))))
    mx.plot(*pose.position[:2], "v", color=INK, ms=6)
    mx.text(pose.position[0] + 3, pose.position[1] - 5, "drone, this frame", fontsize=8, color=INK)
    mx.set_aspect("equal")
    mx.set_xlabel("east (m)", color=INK_2, fontsize=9)
    mx.set_ylabel("north (m)", color=INK_2, fontsize=9)
    counts = {k: sum(f.kind == k for f in findings) for k in KIND_COLOUR}
    handles = [
        Patch(color=KIND_COLOUR[k], label=f"{KIND_LABEL[k]} ({counts[k]})") for k in KIND_COLOUR
    ]
    mx.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    title(mx, f"Defect map: {len(findings)} modules, 4,200 checked", 10.5)
    fig.savefig(DOCS / "hero.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------- poses
def pose_figure(scene, defects):
    true = flown_poses(scene, np.random.default_rng(7))
    twin = ThermalTwin(scene, defects, seed=7)
    idx = [i for i, p in enumerate(true) if p.line == 3]
    imgs = [twin.render(true[i], speed_mps=speeds(scene)[3]) for i in idx]
    logged = logged_poses(true, GNSS, np.random.default_rng(10_001))
    frames = [(img, logged[i]) for img, i in zip(imgs, idx, strict=True)]
    refined, _ = refine_poses(scene, frames)
    _, views = Mapper(scene).run([(img, true[i]) for img, i in zip(imgs, idx, strict=True)])
    seen = {v.module_id for v in views}
    truth = [d for d in defects if d.module_id in seen]
    order = {"offline": 0, "substring": 1, "hot_cells": 2, "hot_cell": 3}
    truth = sorted(truth, key=lambda d: order.get(d.kind, 9))[:5]
    from pvdefect.mapper import Finding

    best = best_views([Finding(d.module_id, d.kind, d.delta_t, 2, 2) for d in truth], views)
    res = json.loads((RESULTS / "main.json").read_text())["runs"]
    n_flights = len({r["seed"] for r in res if r["errors"] == "GNSS 1.5 m"})

    fig = plt.figure(figsize=(13.2, 5.6), facecolor=SURFACE)
    for j, d in enumerate(truth):
        k = best[d.module_id]
        m = scene.module(d.module_id)
        for r, (pose, label) in enumerate(
            ((logged[idx[k]], "logged pose\n(GNSS, no RTK)"), (refined[k], "after\nrefinement"))
        ):
            ax = fig.add_axes([0.085 + j * 0.088, 0.47 - r * 0.43, 0.08, 0.35])
            crop = module_crop(np.asarray(imgs[k]), m, pose, scene, 0.03)
            lo, hi = np.nanpercentile(crop, [2, 99.5])
            ax.imshow(crop, cmap="inferno", vmin=lo, vmax=max(hi, lo + 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(f"{d.module_id}\n{KIND_LABEL[d.kind]}", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK, rotation=0, ha="right", va="center")
    fig.text(
        0.02,
        0.955,
        "Same frames, two poses: the defective modules where each pose says they are",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    names = ["RTK", "attitude 1 deg", "GNSS 1.5 m"]
    labels = ["RTK", "1° heading/\ngimbal bias", "GNSS,\nno RTK"]
    for p, (metric, ylab) in enumerate(
        (("f1", "F1, right module"), ("cell_accuracy", "hot cell on the right cell"))
    ):
        ax = fig.add_axes([0.52 + p * 0.25, 0.22, 0.2, 0.6])
        style(ax)
        x = np.arange(len(names))
        methods = (
            ("logged", "#b5b3ad", "logged poses"),
            ("pnp", ORANGE, "PnP baseline"),
            ("ours", BLUE, "refined (ours)"),
        )
        for o, (method, colour, lab) in enumerate(methods):
            vals = [
                [r[metric] for r in res if r["errors"] == nm and r["method"] == method]
                for nm in names
            ]
            mean = [np.nanmean(v) if np.isfinite(v).any() else 0.0 for v in vals]
            lo = [m - np.nanmin(v, initial=m) for m, v in zip(mean, vals, strict=True)]
            hi = [np.nanmax(v, initial=m) - m for m, v in zip(mean, vals, strict=True)]
            ax.bar(x + (o - 1) * 0.27, mean, 0.25, color=colour, label=lab)
            ax.errorbar(x + (o - 1) * 0.27, mean, [lo, hi], fmt="none", ecolor=INK_2, lw=0.8)
        ax.set_xticks(x, labels, fontsize=7.5, color=INK_2)
        ax.set_ylim(0, 1.05)
        ax.set_title(ylab, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        if p == 0:
            ax.legend(
                frameon=False,
                fontsize=8.5,
                loc="upper center",
                bbox_to_anchor=(1.12, -0.17),
                ncol=3,
            )
    fig.text(
        0.52,
        0.955,
        f"Whole plant, {n_flights} flights per case (bars: mean, whiskers: range)",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(DOCS / "pose_refinement.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------- glare
def reflection_figure(defects):
    scene = Scene.from_mission(MORNING).flown_at("13:00")
    pose = scene.poses[12]
    img = ThermalTwin(scene, defects, seed=1).render(pose)
    views_off = Mapper(scene, Settings(reflection_guard_deg=0.0)).frame_views(0, img, pose)
    ids = {d.module_id for d in defects}
    fig = plt.figure(figsize=(12.6, 5.0), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.04, 0.47, 0.8])
    lo, hi = np.percentile(img, [40, 99.9])  # stretch over the modules, ground goes dark
    ax.imshow(img, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_axis_off()
    for v in views_off:
        if v.kind == "cell" and v.module_id not in ids:
            outline(ax, scene, scene.module(v.module_id), pose, AQUA, lw=1.4)
    ax.set_title(
        "The morning plan flown at 13:00: the sun's reflection crosses the rows",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=AQUA, lw=2, label="healthy module flagged as hot (no guard)")
        ],
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
    )
    res = json.loads((RESULTS / "reflections.json").read_text())["runs"]
    variants = ["single image", "multi-view fusion", "glare guard", "fusion + glare guard"]
    for p, overlap in enumerate(("all images", "every 2nd image")):
        bx = fig.add_axes([0.55 + p * 0.235, 0.2, 0.2, 0.62])
        style(bx)
        vals = [
            next(r["fp"] for r in res if r["plan"].startswith("morning") and r["images"] == overlap
                 and r["variant"] == v)
            for v in variants
        ]  # fmt: skip
        bars = bx.bar(range(4), vals, color=[ORANGE, "#b5b3ad", AQUA, BLUE])
        for b, val in zip(bars, vals, strict=True):
            bx.text(
                b.get_x() + b.get_width() / 2,
                val + 12,
                str(val),
                ha="center",
                fontsize=8.5,
                color=INK,
            )
        ticks = ["single\nimage", "multi-view\nfusion", "glare\nguard", "fusion\n+ guard"]
        bx.set_xticks(range(4), ticks, fontsize=7.5, color=INK_2)
        bx.set_ylim(0, 760)
        sub = "about 3 views per module" if overlap == "all images" else "1-2 views per module"
        bx.set_title(f"{overlap}: {sub}", loc="left", fontsize=9, color=INK, fontweight="bold")
        if p == 0:
            bx.set_ylabel("false alarms", fontsize=8.5, color=INK_2)
    fig.text(
        0.55,
        0.93,
        "False alarms over the whole plant (healthy modules reported; 44 real defects)",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(DOCS / "reflections.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------- raptor
def raptor_figure(dataset: Path | None):
    res = json.loads((RESULTS / "raptor.json").read_text())
    fig = plt.figure(figsize=(12.6, 4.9), facecolor=SURFACE)
    ax = fig.add_axes([0.06, 0.16, 0.36, 0.7])
    style(ax)
    curves = [
        ("Cell: single hot cell", "sweep_hot", "Cell", BLUE),
        ("Cell-Multi: several hot cells", "sweep_hot", "Cell-Multi", AQUA),
        ("Diode: one substring warm (substring test)", "sweep_band", "Diode", ORANGE),
    ]
    for label, key, cls, colour in curves:
        pts = [(r["false_alarm"], r[cls]) for r in res[key] if r["false_alarm"] > 0]
        ax.plot(*zip(*pts, strict=True), "-o", color=colour, ms=3.5, lw=1.8, label=label)
    for r in res["sweep_hot"]:
        if r["hot_k"] in (3, 4, 6):
            ax.annotate(f"{r['hot_k']}σ", (r["false_alarm"], r["Cell"]), (5, -12),
                        textcoords="offset points", fontsize=7.5, color=INK_2)  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_xscale("log")
    ax.set_xlim(8e-5, 0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("flagged No-Anomaly modules (false-alarm rate)", color=INK_2, fontsize=9)
    ax.set_ylabel("anomalous modules flagged", color=INK_2, fontsize=9)
    title(ax, "Real crops: Raptor Maps InfraredSolarModules (20,000)", 10.5)
    if dataset is not None:
        from pvdefect.realdata import classify, load

        imgs, classes = load(dataset)
        classes = np.array(classes)
        st = Settings(hot_k=4.0)
        rng = np.random.default_rng(3)
        for r, cls in enumerate(("Cell", "Diode", "No-Anomaly")):
            ids = rng.choice(np.flatnonzero(classes == cls), 8, replace=False)
            fig.text(0.49, 0.78 - r * 0.28, cls, fontsize=9, color=INK, ha="right", va="center")
            for j, i in enumerate(ids):
                a = fig.add_axes([0.5 + j * 0.061, 0.66 - r * 0.28, 0.05, 0.24])
                a.imshow(imgs[i], cmap="inferno", interpolation="nearest")
                a.set_xticks([])
                a.set_yticks([])
                out = classify(imgs[i], st)
                colour = {"cell": BLUE, "substring": ORANGE, "none": GRID}[out]
                for s in a.spines.values():
                    s.set_color(colour)
                    s.set_linewidth(2.2 if out != "none" else 0.8)
        fig.text(
            0.5,
            0.93,
            "Random examples. Frame: hot spot (blue), substring (orange), not flagged (grey)",
            fontsize=9,
            color=INK,
        )
    fig.savefig(DOCS / "raptor.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raptor", type=Path, help="unzipped InfraredSolarModules folder (example crops)"
    )
    ap.add_argument("--only", choices=["hero", "poses", "reflections", "raptor"])
    a = ap.parse_args()
    scene = Scene.from_mission(NOON)
    defects = random_defects(scene, np.random.default_rng(7))
    if a.only in (None, "raptor"):
        raptor_figure(a.raptor)
    if a.only in (None, "reflections"):
        reflection_figure(defects)
    if a.only in (None, "poses"):
        pose_figure(scene, defects)
    if a.only in (None, "hero"):
        frames, findings, views, report = example_survey(scene, defects)
        from pvdefect.evaluate import score

        sc = score(findings, defects).as_dict()
        out = ROOT / "examples" / "june-noon" / "results"
        write_outputs(out, scene, findings, views, frames, report, sc)
        print(json.dumps(sc))
        hero(scene, frames, findings, views)


if __name__ == "__main__":
    main()
