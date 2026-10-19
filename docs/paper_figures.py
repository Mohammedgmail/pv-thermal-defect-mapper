"""Figures, tables and numbers for the paper from docs/results/*.json.

python docs/paper_figures.py      # docs/paper/*.pdf, *.png, tables/*.tex and numbers.tex

Every number quoted in the paper comes from numbers.tex, so the text cannot drift from the
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pvdefect.errormodel import (  # noqa: E402
    jacobian,
    offset_sd,
    p_right_cell,
    p_same_cell,
    pose_sd,
)
from pvdefect.scene import Scene  # noqa: E402
from pvdefect.simulate import RTK, PoseErrors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "paper"
NOON = ROOT / "examples" / "june-noon" / "mission.json"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
METHOD = {
    "logged": ("logged poses", "#b5b3ad"),
    "pnp": ("PnP baseline", ORANGE),
    "ours": ("ours", BLUE),
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    }
)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------ error model
def fig_error_model():
    """(a) chance of the right cell against attitude error, for three viewing distances,
    with the simulation's measurements; (b) the same for modules against position error."""
    scene = Scene.from_mission(NOON)
    pose, row = scene.poses[80], scene.rows[3]
    m = scene.module("R04-C010-1")
    J = jacobian(pose, scene, row)
    att = np.linspace(0.0, 1.2, 121)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ax = axes[0]
    style(ax)
    for scale, ls in ((0.7, ":"), (1.0, "-"), (1.4, "--")):
        p = [
            p_right_cell(
                offset_sd(
                    J * [1, 1, 1, scale, scale],
                    np.radians([0, 0, 0, a, a]) * [0, 0, 0, 1, 1] + [0.02, 0.02, 0.03, 0, 0],
                ),
                m.cell_size,
            )
            for a in att
        ]
        rng_m = np.linalg.norm(J[:, 3]) * scale
        ax.plot(att, p, ls, color=BLUE, lw=1.3, label=f"{rng_m:.0f} m horizontal range")
    ax.set_xlabel("heading and gimbal-pitch error, s.d. (deg)")
    ax.set_ylabel("hot cell on the right cell")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) Cell level: attitude dominates", loc="left", fontweight="bold")
    main = load("main")
    if main:
        for name, mk in (("RTK", "o"), ("attitude 1 deg", "s")):
            runs = [r for r in main["runs"] if r["errors"] == name and r["method"] == "logged"]
            e = (
                RTK
                if name == "RTK"
                else PoseErrors(
                    yaw_bias_deg=1, yaw_jitter_deg=0.5, pitch_bias_deg=1, pitch_jitter_deg=0.5
                )
            )
            sd = float(np.degrees(np.hypot(*pose_sd(e)[3:]) / np.sqrt(2)))
            acc = [r["cell_accuracy"] for r in runs if r["cell_accuracy"] == r["cell_accuracy"]]
            if acc:
                ax.errorbar(
                    sd,
                    np.mean(acc),
                    yerr=[[np.mean(acc) - min(acc)], [max(acc) - np.mean(acc)]],
                    fmt=mk,
                    color=ORANGE,
                    ms=4,
                    capsize=2,
                    label=f"simulation, {name} (logged)",
                )
            runs = [r for r in main["runs"] if r["errors"] == name and r["method"] == "ours"]
            acc = [r["cell_accuracy"] for r in runs if r["cell_accuracy"] == r["cell_accuracy"]]
            if acc:
                ax.plot(
                    sd,
                    np.mean(acc),
                    mk,
                    color=AQUA,
                    ms=4,
                    mfc="none",
                    label=f"simulation, {name} (refined)",
                )
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    style(ax)
    pos = np.linspace(0, 3.0, 151)
    p_mod = [
        p_right_cell(
            offset_sd(J, [x, x, 1.5 * x, np.radians(0.36), np.radians(0.28)]), (m.width, m.height)
        )
        for x in pos
    ]
    p_cell = [
        p_right_cell(offset_sd(J, [x, x, 1.5 * x, np.radians(0.36), np.radians(0.28)]), m.cell_size)
        for x in pos
    ]
    ax.plot(pos, p_mod, color=BLUE, lw=1.3, label="right module")
    ax.plot(pos, p_cell, color=BLUE, lw=1.3, ls="--", label="right cell")
    ax.axvspan(0, 0.05, color=AQUA, alpha=0.15, lw=0)
    ax.text(0.07, 0.08, "RTK", color=INK_2, fontsize=7)
    ax.axvspan(1.0, 2.5, color=ORANGE, alpha=0.1, lw=0)
    ax.text(1.05, 0.08, "GNSS without RTK", color=INK_2, fontsize=7)
    ax.set_xlabel("horizontal position error, s.d. (m)")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) Module level: position dominates", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "error_model")


def requirement_table():
    """Pose accuracy needed for 95 % right cells / modules from logged poses, per axis."""
    scene = Scene.from_mission(NOON)
    pose, row = scene.poses[80], scene.rows[3]
    m = scene.module("R04-C010-1")
    J = jacobian(pose, scene, row)
    out = {}
    for what, size in (("cell", m.cell_size), ("module", (m.width, m.height))):
        for axis, j in (("yaw_deg", 3), ("pitch_deg", 4), ("position_m", 0)):
            for x in np.logspace(-4, 1, 2000):
                sd = np.zeros(5)
                if axis == "position_m":
                    sd[:3] = [x, x, 1.5 * x]
                else:
                    sd[j] = np.radians(x)
                if p_right_cell(offset_sd(J, sd), size) < 0.95:
                    out[f"{what}_{axis}"] = round(float(x), 4)
                    break
    print(json.dumps(out, indent=1))
    (RESULTS / "requirements.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------ helpers
def ci95(x) -> tuple[float, float, float]:
    """Mean and a 95 % bootstrap interval (2,000 resamples, fixed seed)."""
    x = np.asarray([v for v in x if v == v], float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(0)
    boot = rng.choice(x, (2000, x.size)).mean(axis=1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fmt(x, digits=2) -> str:
    return "--" if x != x else f"{x:.{digits}f}"


def pct(x, digits=0) -> str:
    return "--" if x != x else f"{100 * x:.{digits}f}\\,\\%"


NUMBERS: dict[str, str] = {}


def number(name: str, value: str) -> None:
    """A macro for numbers.tex: \\name{} in the text."""
    NUMBERS[name] = value


def write_numbers() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by docs/paper_figures.py from docs/results/*.json; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUMBERS.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "numbers.tex", f"({len(NUMBERS)} numbers)")


def table(name: str, header: list[str], rows: list[list[str]], align: str) -> None:
    """A booktabs tabular for \\input{} in the paper."""
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    body = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\"]
    body.append("\\midrule")
    body += [" & ".join(r) + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(body) + "\n")
    print("wrote", OUT / "tables" / f"{name}.tex")


def macro(*parts) -> str:
    """CamelCase macro name from parts, letters only (TeX macro names)."""
    out = ""
    for p in parts:
        text = str(p)
        for sep in "-_.+":
            text = text.replace(sep, " ")
        for w in text.split():
            w = "".join(c for c in w if c.isalpha() or c.isdigit())
            w = w.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three")
            w = (
                w.replace("4", "Four")
                .replace("5", "Five")
                .replace("6", "Six")
                .replace("7", "Seven")
            )
            w = w.replace("8", "Eight").replace("9", "Nine")
            out += w[:1].upper() + w[1:]
    return out


# ------------------------------------------------------------------------------ main results
ERR_LABEL = {
    "RTK": "RTK",
    "attitude 1 deg": "RTK, attitude off by 1\\textdegree",
    "GNSS 1.5 m": "GNSS without RTK (1.5\\,m)",
}


def main_table():
    main = load("main")
    if not main:
        return
    rows = []
    for err, label in ERR_LABEL.items():
        for method in ("logged", "pnp", "ours"):
            runs = [r for r in main["runs"] if r["errors"] == err and r["method"] == method]
            if not runs:
                continue
            f1 = ci95([r["f1"] for r in runs])
            cell = ci95([r["cell_accuracy"] for r in runs])
            fp = ci95([r["fp"] for r in runs])
            pose = float(np.median([r["pose_err_median_m"] for r in runs]))
            perfect = sum(r["f1"] >= 0.995 for r in runs)
            rows.append(
                [
                    label if method == "logged" else "",
                    METHOD[method][0],
                    f"{fmt(f1[0])} [{fmt(f1[1])}, {fmt(f1[2])}]",
                    f"{perfect}/{len(runs)}",
                    f"{fmt(fp[0], 1)}",
                    pct(cell[0]),
                    f"{100 * pose:.1f}",
                ]
            )
            key = macro("main", err, method)
            number(key + "Fone", fmt(f1[0]))
            number(key + "FoneLo", fmt(f1[1]))
            number(key + "FoneHi", fmt(f1[2]))
            number(key + "Cell", pct(cell[0]))
            number(key + "Fp", fmt(fp[0], 1))
            number(key + "PoseCm", f"{100 * pose:.1f}")
            number(key + "Runs", str(len(runs)))
            number(key + "Perfect", str(perfect))
            number(key + "FoneMedian", fmt(float(np.median([r["f1"] for r in runs]))))
            if method == "ours":
                number(
                    key + "SecPerFrame", fmt(np.mean([r["correction_s_per_frame"] for r in runs]))
                )
    table(
        "main",
        [
            "Logged poses",
            "Method",
            "F1 [95\\,\\% CI]",
            "F1\\,=\\,1",
            "False alarms",
            "Exact cell",
            "Pose (cm)",
        ],
        rows,
        "llrrrrr",
    )
    number("mainDefects", str(main.get("defects", "")))
    number("mainFrames", str(main.get("frames", "")))


def ablation_table():
    abl = load("ablation")
    if not abl:
        return
    names = {
        "ours:t": "across-row offset only",
        "ours:t+scale": "+ distance to the module plane",
        "ours:t+scale+s": "+ along-row offset (modulo a module)",
        "ours:t+scale+s+anchors": "+ row-end anchors",
        "ours:full": "+ texture links between frames (full)",
    }
    rows = []
    for m, label in names.items():
        cells = [label]
        for err in ("attitude 1 deg", "GNSS 1.5 m"):
            runs = [r for r in abl["runs"] if r["errors"] == err and r["method"] == m]
            f1 = ci95([r["f1"] for r in runs])
            cell = ci95([r["cell_accuracy"] for r in runs])
            cells += [fmt(f1[0]), pct(cell[0])]
            number(macro("abl", m.removeprefix("ours:"), err) + "Fone", fmt(f1[0]))
        rows.append(cells)
    table(
        "ablation",
        ["Refinement steps", "F1 (att.)", "Cell (att.)", "F1 (GNSS)", "Cell (GNSS)"],
        rows,
        "lrrrr",
    )


# ------------------------------------------------------------------------------ stress
def fig_stress():
    st = load("stress")
    if not st:
        return
    runs = st["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, kind, xlabel in (
        (axes[0], "gnss_bias", "position bias (m), each axis"),
        (axes[1], "attitude_bias", "heading and gimbal bias (deg)"),
    ):
        style(ax)
        for method in ("logged", "pnp", "ours"):
            sel = [r for r in runs if r["experiment"] == kind and r["method"] == method]
            xs = sorted({r["value"] for r in sel})
            ys = [np.mean([r["f1"] for r in sel if r["value"] == x]) for x in xs]
            lab, col = METHOD[method]
            ax.plot(xs, ys, "o-", color=col, ms=3, lw=1.3, label=lab)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("F1 (module level)")
    axes[0].set_title("(a) Position bias", loc="left", fontweight="bold")
    axes[1].set_title("(b) Attitude bias", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower left")
    ax = axes[2]
    style(ax)
    sel = [r for r in runs if r["experiment"] == "detection_limit" and "recall_by_dt" in r]
    for r in sel:
        xs = [float(k) for k in r["recall_by_dt"]]
        ys = list(r["recall_by_dt"].values())
        ax.plot(xs, ys, "o-", ms=3, lw=1.2, label=f"{r['hot_k']:g} K ({r['fp']} false alarms)")
    ax.set_xlabel("hot-cell excess temperature (K)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(c) Detection limit", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "stress")
    # numbers: largest bias with F1 >= 0.95 for each method
    for kind in ("gnss_bias", "attitude_bias"):
        for method in ("logged", "pnp", "ours"):
            sel = [r for r in runs if r["experiment"] == kind and r["method"] == method]
            ok = [
                v
                for v in sorted({r["value"] for r in sel})
                if np.mean([r["f1"] for r in sel if r["value"] == v]) >= 0.95
            ]
            number(macro("stress", kind, method, "max"), f"{max(ok):g}" if ok else "0")
    image_rows = []
    for kind, label, unit in (
        ("frame_contrast", "Frame-line contrast", "K"),
        ("netd", "Sensor noise (NETD)", "K"),
        ("speed", "Flight speed", "$\\times$ plan"),
    ):
        for v in sorted({r["value"] for r in runs if r["experiment"] == kind}):
            sel = [r for r in runs if r["experiment"] == kind and r["value"] == v]
            exact = [r for r in sel if r.get("poses") == "exact"]
            ours = [r for r in sel if r.get("poses") == "GNSS 1.5 m" and r["method"] == "ours"]
            image_rows.append(
                [
                    label,
                    f"{v:g} {unit}",
                    fmt(np.mean([r["f1"] for r in exact])) if exact else "--",
                    fmt(np.mean([r["f1"] for r in ours])) if ours else "--",
                    fmt(np.mean([r["pose_err_median_m"] for r in ours]) * 100, 1) if ours else "--",
                ]
            )
    table(
        "stress_image",
        ["Condition", "Value", "F1, exact poses", "F1, GNSS + ours", "Pose (cm)"],
        image_rows,
        "llrrr",
    )

    def mean_of(kind, value, method, key, poses=None):
        sel = [
            r[key]
            for r in runs
            if r["experiment"] == kind
            and r["value"] == value
            and r["method"] == method
            and (poses is None or r.get("poses") == poses)
            and r[key] == r[key]
        ]
        return float(np.mean(sel)) if sel else float("nan")

    for method in ("logged", "pnp", "ours"):
        number(macro("stress cell att one", method), pct(mean_of("attitude_bias", 1.0, method,
                                                                  "cell_accuracy")))  # fmt: skip
    for v in (0.5, 1.5):
        number(macro("stress no ends", v, "ours") + "Fone", fmt(mean_of("no_row_ends", v, "ours",
                                                                       "f1")))  # fmt: skip
    for v, att in ((2.0, "two"), (3.0, "three")):
        number(macro("stress cell att", att, "ours"), pct(mean_of("attitude_bias", v, "ours",
                                                                  "cell_accuracy")))  # fmt: skip
    low = [
        r["f1"]
        for r in runs
        if r["experiment"] == "frame_contrast" and r["value"] == -0.5 and r["method"] == "ours"
    ]
    if low:
        number("stressFaintLinesFoneMin", fmt(min(low)))
        number("stressFaintLinesFoneMax", fmt(max(low)))
    det = {r["hot_k"]: r for r in runs if r["experiment"] == "detection_limit"}
    if det:
        number("detOneFiveRecallTwo", pct(det[1.5]["recall_by_dt"]["2"]))
        number("detOneFiveFp", str(det[1.5]["fp"]))
        number("detThreeRecallFour", pct(det[3.0]["recall_by_dt"]["4"]))
        number("detOneFp", str(det[1.0]["fp"]))
    rough = {r["hot_k"]: r for r in runs if r["experiment"] == "non_uniformity"
             and r["value"] == "rough"}  # fmt: skip
    if rough:
        for k, r in rough.items():
            number(macro("rough k", k) + "Fp", str(r["fp"]))
    for kind in ("soiling",):
        sel = [r for r in runs if r["experiment"] == kind]
        if sel:
            number("soilingFp", str(int(np.mean([r.get("fp_on_soiled", 0) for r in sel]))))
            number("soilingNuisances", "30")
    for v in sorted({r["value"] for r in runs if r["experiment"] == "non_uniformity"}, key=str):
        sel = [r for r in runs if r["experiment"] == "non_uniformity" and r["value"] == v]
        for r in sel:
            number(macro("nonuni", v, "k", r["hot_k"]) + "Fone", fmt(r["f1"]))


# ------------------------------------------------------------------------------ sites
def sites_table():
    sites = load("sites")
    if not sites:
        return
    facing = {"seville": "south", "gujarat": "south", "queensland": "north"}
    rows = []
    for name in ("seville", "gujarat", "queensland"):
        sel = [r for r in sites["runs"] if r["site"] == name]
        if not sel:
            continue
        info = sel[0]
        cells = [
            name.capitalize(),
            "portrait" if info["portrait"] else "landscape",
            facing[name],
            str(info["rows_per_line"]),
            str(info["frames"]),
        ]
        for err in ("RTK", "GNSS 1.5 m"):
            vals = []
            for method in ("logged", "pnp", "ours"):
                runs = [r for r in sel if r["errors"] == err and r["method"] == method]
                f1 = ci95([r["f1"] for r in runs])
                vals.append(fmt(f1[0]))
                number(macro("site", name, err, method) + "Fone", fmt(f1[0]))
            cells.append(" / ".join(vals[:2]) + f" / \\textbf{{{vals[2]}}}")
        cell = ci95(
            [r["cell_accuracy"] for r in sel if r["method"] == "ours" and r["errors"] != "none"]
        )
        cells.append(pct(cell[0]))
        rows.append(cells)
    table(
        "sites",
        [
            "Plant",
            "Modules",
            "Facing",
            "Rows/pass",
            "Frames",
            "F1, RTK",
            "F1, GNSS",
            "Exact cell",
        ],
        rows,
        "lllrrccr",
    )


# ------------------------------------------------------------------------------ reflections etc.
def reflections_table():
    ref = load("reflections")
    if not ref:
        return
    rows = []
    for variant in ("single image", "multi-view fusion", "glare guard", "fusion + glare guard"):
        cells = [variant]
        for plan, images in (
            ("morning plan at 13:00", "all images"),
            ("morning plan at 13:00", "every 2nd image"),
            ("noon plan at 13:00", "all images"),
        ):
            r = next(
                x
                for x in ref["runs"]
                if x["variant"] == variant and x["plan"] == plan and x["images"] == images
            )
            cells.append(f"{r['fp']} / {fmt(r['f1'])}")
            number(macro("refl", plan, images, variant) + "Fp", str(r["fp"]))
        rows.append(cells)
    table(
        "reflections",
        ["Method", "Morning, all", "Morning, every 2nd", "Noon, all"],
        rows,
        "lrrr",
    )
    for r in ref["runs"]:
        number(macro("refl", r["plan"], "glare"), f"{r['min_glare_deg']:g}")


def reinspection_numbers():
    ri = load("reinspection")
    if not ri:
        return
    for r in ri["runs"]:
        key = macro("reinsp", r["plan"])
        number(key + "Candidates", str(len(r["candidates"])))
        number(key + "Minutes", fmt(r["reflight_minutes"], 1))
        number(key + "FoneBefore", fmt(r["survey"]["f1"], 3))
        if "after" in r:
            number(key + "FoneAfter", fmt(r["after"]["f1"], 3))
    sitl = sorted(RESULTS.glob("reinspection/*.sitl.json"))
    for path in sitl:
        x = json.loads(path.read_text())
        key = macro("sitl", path.stem.split(".")[0])
        number(key + "Reached", f"{x['viewpoints_reached_2m']}/{x['viewpoints']}")
        number(key + "Images", str(x["images"]))
        if x.get("flight_minutes_sitl") is not None:
            number(key + "Minutes", fmt(x["flight_minutes_sitl"], 1))


def pvhawk_numbers():
    pv = load("pvhawk")
    if not pv:
        return
    for m, r in pv["summary"].items():
        key = macro("hawk", m)
        number(key + "Share", pct(r["share_right"], 1))
        number(key + "Right", str(r["right"]))
        number(key + "Corner", fmt(r["corner_px_median"], 1))
        number(key + "FramesAllRight", str(r["frames_all_right"]))
    number("hawkModules", str(pv["summary"]["logged"]["modules"]))
    number("hawkFrames", str(pv["frames"]))
    for h, summ in pv["height_sensitivity"].items():
        number(macro("hawk height", h, "ours") + "Share", pct(summ["ours"]["share_right"], 1))
    rows = []
    for m in ("logged", "PnP baseline", "ours, no row ends", "ours, no texture links", "ours"):
        r = pv["summary"][m]
        rows.append(
            [
                m,
                f"{r['right']}/{r['modules']}",
                pct(r["share_right"], 1),
                fmt(r["corner_px_median"], 1),
            ]
        )
    r = pv["summary"]["SfM poses"]
    rows.append(
        [
            "SfM poses (reference, 19 frames)",
            f"{r['right']}/{r['modules']}",
            pct(r["share_right"], 1),
            fmt(r["corner_px_median"], 1),
        ]
    )
    table("pvhawk", ["Poses", "Right / seen", "Share", "Corner error (px)"], rows, "lrrr")


def raptor_numbers():
    rp = load("raptor")
    if not rp:
        return
    d = rp["default"]
    number("raptorImages", f"{rp['images']:,}".replace(",", "{,}"))
    for cls, key in (("Cell", "cell"), ("Cell-Multi", "cell"), ("Diode", "substring")):
        number(macro("raptor", cls, "flagged"), pct(d[cls][key]))
    for cls in ("Cracking", "Soiling", "Vegetation"):
        number(macro("raptor", cls, "hot"), pct(d[cls]["cell"]))
    number("raptorCleanNotFlagged", pct(d["No-Anomaly"]["none"], 1))
    number("raptorCleanSubstring", pct(d["No-Anomaly"]["substring"], 1))
    four = next(x for x in rp["sweep_hot"] if x["hot_k"] == 4)
    number("raptorFourFalseAlarm", pct(four["false_alarm"], 1))
    number("raptorFourCell", pct(four["Cell"]))
    number("raptorFourCellMulti", pct(four["Cell-Multi"]))


def requirement_numbers():
    req = load("requirements")
    if not req:
        return
    for k, v in req.items():
        unit = 1000 if k.endswith("position_m") else 1
        number(macro("req", k), f"{v * unit:.3g}")


# ------------------------------------------------------------------------------ re-inspection
def fig_reinspection():
    """(a) Map of the re-inspection flight flown in SITL; (b) the module in the survey view
    whose hot spot was dropped as a reflection; (c) the module in the re-inspection image."""
    import importlib.util

    import pandas as pd

    from pvdefect import reinspect as RI
    from pvdefect.mapper import Mapper, crop_grid, module_crop
    from pvdefect.simulate import ThermalTwin, random_defects

    sitl_path = RESULTS / "reinspection" / "morning_plan_at_1300_step2.sitl.json"
    if not sitl_path.exists():
        return
    sitl = json.loads(sitl_path.read_text())
    spec = importlib.util.spec_from_file_location("rx", ROOT / "docs" / "run_experiments.py")
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    noon = Scene.from_mission(NOON)
    defects = random_defects(noon, np.random.default_rng(7))
    scene = Scene.from_mission(ROOT / "examples" / "june-morning" / "mission.json").flown_at(
        "13:00"
    )
    sv = rx.survey(scene, defects)
    frames = sv.frames(sv.true)[::2]
    mapper = Mapper(scene)
    findings, views = mapper.run(frames)
    todo = RI.candidates(findings, views)
    hlat, hlon, _ = scene.extras["mission"]["home_lla"]
    home = np.array([*scene.frame.to_local(hlat, hlon), 0.0])
    plan = RI.plan(scene, todo, sv.true[-1].time + pd.Timedelta(minutes=5), home)
    target = todo[0].module_id
    m = scene.module(target)
    masked = [v for v in views if v.module_id == target and v.reflections]
    k = (
        masked[0].frame
        if masked
        else min(range(len(frames)), key=lambda j: np.linalg.norm(frames[j][1].position - m.center))
    )
    survey_img, survey_pose = frames[k]
    pose = RI.poses(plan)[0]
    scene2 = RI.scene_with(scene, plan)
    re_img = ThermalTwin(scene2, defects, seed=7).render(pose)
    d = next(x for x in defects if x.module_id == target)

    fig = plt.figure(figsize=(7.0, 2.6))
    ax = fig.add_axes([0.0, 0.1, 0.5, 0.8])
    style(ax)
    for r in scene.rows:
        c = r.corners()[:, :2]
        ax.fill(c[:, 0], c[:, 1], color="#d8d6d0", lw=0)
    mc = m.corners()[:, :2]
    ax.fill(mc[:, 0], mc[:, 1], color=ORANGE, lw=0, zorder=3)
    tr = np.array(sitl["track_enu"])
    ax.plot(tr[:, 0], tr[:, 1], color=INK, lw=0.9, label="SITL track", zorder=4)
    for t in sitl["targets_enu"]:
        ax.plot(t[0], t[1], "^", color=BLUE, ms=6, zorder=5, label="viewpoint")
    yaw = np.radians(pose.yaw_deg)
    ax.annotate(
        "",
        xy=(pose.position[0] + 8 * np.sin(yaw), pose.position[1] + 8 * np.cos(yaw)),
        xytext=(pose.position[0], pose.position[1]),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0},
        zorder=5,
    )
    for sh in sitl["shots_enu"]:
        ax.plot(sh[0], sh[1], "o", color=AQUA, ms=3.5, zorder=6, label="image taken")
    ax.plot(home[0], home[1], "s", color=INK, ms=5, label="home", zorder=5)
    xs = [home[0], m.center[0], pose.position[0]]
    ys = [home[1], m.center[1], pose.position[1]]
    ax.set_xlim(min(xs) - 25, max(xs) + 25)
    ax.set_ylim(min(ys) - 15, max(ys) + 15)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h, strict=True))
    ax.legend(
        uniq.values(), uniq.keys(), fontsize=6, loc="upper right", framealpha=0.9, edgecolor=GRID
    )
    ax.set_title(
        f"(a) Re-flight in SITL: {sitl['flight_minutes_sitl']:.1f} min",
        loc="left",
        fontweight="bold",
    )
    for j, (img, ps, title) in enumerate(
        (
            (survey_img, survey_pose, "(b) Survey"),
            (re_img, pose, "(c) Re-inspection"),
        )
    ):
        bx = fig.add_axes([0.55 + j * 0.23, 0.12, 0.2, 0.74])
        crop = module_crop(np.asarray(img), m, ps, scene2, 0.03)
        if crop is None:
            continue
        lo, hi = np.nanpercentile(crop, [2, 99.8])
        bx.imshow(crop, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ms, mt = crop_grid(m, 0.03)
        P = m.point(ms, mt)
        ray = P - ps.position
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        n = m.row.normal
        mirrored = ray - 2 * (ray @ n)[..., None] * n
        sun = mapper._sun(ps.time)
        glare = np.degrees(np.arccos(np.clip(mirrored @ sun, -1, 1)))
        cs = bx.contour(glare, levels=[6.0], colors=[AQUA], linewidths=1.0)
        bx.clabel(cs, fmt="6°", fontsize=5.5)
        bx.text(
            0.02, -0.06, f"sun's reflection {glare.min():.0f}° away", transform=bx.transAxes,
            fontsize=6, color=INK_2, va="top",
        )  # fmt: skip
        bx.set_xticks([])
        bx.set_yticks([])
        bx.set_title(title, loc="left", fontweight="bold")
    fig.text(0.55, 0.93, f"{target}, hot cell +{d.delta_t:.0f} K", fontsize=7, color=INK_2)
    save(fig, "reinspection")


if __name__ == "__main__":
    fig_error_model()
    requirement_table()
    requirement_numbers()
    main_table()
    ablation_table()
    fig_stress()
    sites_table()
    reflections_table()
    reinspection_numbers()
    pvhawk_numbers()
    raptor_numbers()
    fig_reinspection()
    write_numbers()
    _ = p_same_cell
