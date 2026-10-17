"""Write the mapper's results: findings table, GeoJSON of module outlines, crop gallery, report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .mapper import Finding, View, findings_table, module_crop  # noqa: E402
from .scene import CELLS_ACROSS, CELLS_ALONG, FRAME_M, Pose, Scene  # noqa: E402

KIND_LABEL = {
    "hot_cell": "hot cell",
    "hot_cells": "hot cells",
    "substring": "bypass-diode substring",
    "offline": "offline module",
    "string": "offline string",
}
PRIORITY = {"string": 0, "offline": 1, "substring": 2, "hot_cells": 3, "hot_cell": 4}


def ranked(findings: list[Finding]) -> list[Finding]:
    """Most urgent first: strings and offline modules lose most energy, then by temperature."""

    def key(f: Finding):  # an offline string is listed module by module, in order
        return (PRIORITY.get(f.kind, 9), 0.0 if f.kind == "string" else -f.delta_t, f.module_id)

    return sorted(findings, key=key)


def geojson(scene: Scene, findings: list[Finding]) -> dict:
    """Module outlines (WGS84) with the finding as properties, for QGIS or a web map."""
    feats = []
    for f in findings:
        c = scene.module(f.module_id).corners()
        lat, lon = scene.frame.to_geodetic(c[:, 0], c[:, 1])
        ring = [[round(float(x), 8), round(float(y), 8)] for x, y in zip(lon, lat, strict=True)]
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]},
                "properties": {
                    "module": f.module_id,
                    "kind": f.kind,
                    "delta_t_k": f.delta_t,
                    "cells": [list(x) for x in f.cells],
                    "substring": f.substring,
                    "views": f"{f.views_flagged}/{f.views_seen}",
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def best_views(findings: list[Finding], views: list[View]) -> dict[str, int]:
    """Frame index of the clearest agreeing view of each finding (largest temperature rise)."""
    want = {
        "hot_cell": "cell",
        "hot_cells": "cell",
        "substring": "substring",
        "offline": "offline",
        "string": "offline",
    }
    kinds = {f.module_id: want[f.kind] for f in findings}
    best: dict[str, tuple[float, int]] = {}
    for v in views:
        if kinds.get(v.module_id) == v.kind and v.delta_t > best.get(v.module_id, (-np.inf, 0))[0]:
            best[v.module_id] = (v.delta_t, v.frame)
    return {m: fr for m, (_, fr) in best.items()}


def gallery(
    path: Path,
    scene: Scene,
    findings: list[Finding],
    views: list[View],
    frames: list[tuple[np.ndarray, Pose]],
    max_items: int = 40,
    res: float = 0.03,
) -> None:
    """Rectified crop of each finding from its clearest frame, with the flagged cells boxed."""
    items = ranked(findings)[:max_items]
    best = best_views(items, views)
    cols = min(10, max(1, len(items)))
    rows = int(np.ceil(len(items) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 3.25 * rows), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), items, strict=False):
        m = scene.module(f.module_id)
        k = best.get(f.module_id)
        if k is None:
            continue
        img, pose = frames[k]
        crop = module_crop(np.asarray(img), m, pose, scene, res)
        if crop is None:
            continue
        lo, hi = np.nanpercentile(crop, [2, 99.5])
        ax.imshow(crop, cmap="inferno", vmin=lo, vmax=max(hi, lo + 1), interpolation="nearest")
        cw = (m.width - 2 * FRAME_M) / CELLS_ACROSS / res
        ch = (m.height - 2 * FRAME_M) / CELLS_ALONG / res
        ny = crop.shape[0]
        for r, c in f.cells:  # cell row 0 is at the bottom of the module, crop row 0 at the top
            x0, y1 = FRAME_M / res + c * cw, ny - FRAME_M / res - r * ch
            ax.add_patch(
                Rectangle((x0 - 0.5, y1 - ch - 0.5), cw, ch, fill=False, ec="#2a78d6", lw=1.2)
            )
        ax.set_title(
            f"{f.module_id}\n{KIND_LABEL.get(f.kind, f.kind)}\n+{f.delta_t:.1f} K",
            fontsize=6.5,
            color="#0b0b0b",
        )
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _pct(text: str, x: float) -> str:
    return "" if x is None or x != x else text.format(f"{x:.0%}")


def markdown(
    scene: Scene,
    findings: list[Finding],
    n_frames: int,
    refine: list[dict] | None = None,
    score: dict | None = None,
) -> str:
    counts = Counter(f.kind for f in findings)
    lines = [
        f"# Thermal defect map: {scene.name}",
        "",
        f"- Frames analysed: {n_frames}",
        f"- Modules with a finding: {len(findings)} of {scene.n_modules}",
        "",
    ]
    lines += ["| Finding | Modules |", "|---|---:|"]
    lines += [f"| {KIND_LABEL[k]} | {counts[k]} |" for k in PRIORITY if counts.get(k)]
    if refine is not None:
        done = [r for r in refine if r["applied"]]
        lines += [
            "",
            f"Pose refinement: {len(done)} of {len(refine)} passes corrected, median shift "
            f"{np.median([r['shift_m'] for r in done]) if done else 0:.2f} m at the modules.",
        ]
        skipped = [str(r["line"]) for r in refine if not r["applied"]]
        if skipped:
            lines += [
                "Not corrected (table edge not found, logged poses kept): "
                f"line(s) {', '.join(skipped)}."
            ]
    if score is not None:
        lines += [
            "",
            "## Against the simulation's ground truth",
            "",
            f"Precision {score['precision']:.3f}, recall {score['recall']:.3f}, "
            f"F1 {score['f1']:.3f} "
            f"({score['tp']} found, {score['fp']} false, {score['fn']} missed)."
            + _pct(" Defect type right for {} of the found modules.", score["kind_accuracy"])
            + _pct(" Single hot cells located to the exact cell: {}.", score["cell_accuracy"]),
        ]
    lines += [
        "",
        "## Findings, most urgent first",
        "",
        "| Module | Finding | dT (K) | Cells (row, col) | Views | Lat | Lon |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for f in ranked(findings):
        cells = (
            " ".join(f"({r},{c})" for r, c in f.cells)
            if f.cells
            else (f"substring {f.substring + 1}" if f.substring >= 0 else "")
        )
        lines.append(
            f"| {f.module_id} | {KIND_LABEL.get(f.kind, f.kind)} | {f.delta_t:.1f} | {cells} | "
            f"{f.views_flagged}/{f.views_seen} | {f.lat:.6f} | {f.lon:.6f} |"
        )
    lines += [
        "",
        "Module IDs: R = row, C = column, level 1 = lower module. Columns (of modules and "
        "of cells, 0-5) count from the row's east end for a south-facing row (in general, from "
        "the end 90 degrees anticlockwise of the facing direction); cell rows count from the "
        "bottom (0-11).",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    out: Path,
    scene: Scene,
    findings: list[Finding],
    views: list[View],
    frames: list[tuple[np.ndarray, Pose]],
    refine: list[dict] | None = None,
    score: dict | None = None,
) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = findings_table(ranked(findings))
    paths = [out / "findings.csv", out / "findings.geojson", out / "report.md", out / "gallery.png"]
    df.to_csv(paths[0], index=False)
    paths[1].write_text(json.dumps(geojson(scene, findings)))
    paths[2].write_text(markdown(scene, findings, len(frames), refine, score))
    gallery(paths[3], scene, findings, views, frames)
    if refine is not None:
        (out / "refine.json").write_text(json.dumps(refine, indent=1))
        paths.append(out / "refine.json")
    if score is not None:
        (out / "score.json").write_text(json.dumps(score, indent=1))
        paths.append(out / "score.json")
    return paths
