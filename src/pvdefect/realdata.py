"""Check the per-module analyser on real thermal crops: Raptor Maps InfraredSolarModules.

The dataset (MIT licence, https://github.com/RaptorMaps/InfraredSolarModules) has 20,000 module
crops of 24 x 40 pixels from operating plants, 8-bit and contrast-stretched per image, each
labelled with one of 12 classes. There are no temperatures and no poses, so this checks only
the part of the pipeline that looks inside one module: hot spots and bypass-diode substrings,
with thresholds in robust standard deviations (``relative=True``). Offline modules need row
neighbours and are not tested here.

    pvdefect raptor path/to/InfraredSolarModules
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from pvmission.plant import Row
from scipy import ndimage

from .mapper import Settings, analyse_crop, crop_grid
from .scene import Module

HOT = ("Cell", "Cell-Multi", "Hot-Spot", "Hot-Spot-Multi")
DIODE = ("Diode", "Diode-Multi")
NONE = ("No-Anomaly",)
OTHER = ("Cracking", "Shadowing", "Vegetation", "Soiling", "Offline-Module")

# a 72-cell module in portrait (2.278 m x 1.134 m) to resample the crops onto
_ROW = Row(0, np.zeros(3), 1.134, 2.278, 25.0, 180.0, 1, 1)
MODULE = Module(_ROW, 1, 1)


def load(folder: Path) -> tuple[np.ndarray, list[str]]:
    """Images (N, 40, 24) as float and their class names, from the unzipped dataset folder."""
    folder = Path(folder)
    meta_path = next(folder.rglob("module_metadata.json"))
    meta = json.loads(meta_path.read_text())
    keys = sorted(meta, key=int)
    imgs = np.stack(
        [
            np.asarray(
                Image.open(meta_path.parent / meta[k]["image_filepath"]).convert("L"),
                dtype=np.float32,
            )
            for k in keys
        ]
    )
    return imgs, [meta[k]["anomaly_class"] for k in keys]


def classify(img: np.ndarray, settings: Settings, trim: int = 2) -> str:
    """'cell', 'substring' or 'none' for one crop. The crops include the module frame and a
    little background, cut off by ``trim`` pixels on each side."""
    if trim:
        img = img[trim:-trim, trim:-trim]
    ms, _ = crop_grid(MODULE, settings.resolution_m)
    crop = ndimage.zoom(img, (ms.shape[0] / img.shape[0], ms.shape[1] / img.shape[1]), order=1)
    a = analyse_crop(crop, MODULE, settings, settings.resolution_m, relative=True)
    return "substring" if a["substring"] >= 0 else ("cell" if a["blobs"] else "none")


def evaluate(
    folder: Path,
    settings: Settings = Settings(),
    hot_k=(2, 3, 4, 5, 6, 8, 10),
    band_k=(1.0, 1.5, 2.0, 3.0, 4.0),
    per_class: int | None = None,
    seed: int = 0,
) -> dict:
    """Flag rates per class at the default thresholds, plus threshold sweeps.

    ``per_class`` caps the images used per class (random sample) for a quicker run.
    """
    imgs, classes = load(folder)
    idx = np.arange(len(classes))
    if per_class:
        rng = np.random.default_rng(seed)
        keep = []
        for c in sorted(set(classes)):
            ids = idx[np.array(classes) == c]
            keep += list(rng.choice(ids, min(per_class, len(ids)), replace=False))
        idx = np.array(sorted(keep))
    cls = [classes[i] for i in idx]

    def rates(st: Settings) -> dict[str, dict[str, float]]:
        out: dict[str, Counter] = {}
        for i, c in zip(idx, cls, strict=True):
            out.setdefault(c, Counter())[classify(imgs[i], st)] += 1
        return {
            c: {k: v / sum(n.values()) for k, v in n.items()} | {"n": sum(n.values())}
            for c, n in sorted(out.items())
        }

    default = rates(settings)
    sweep_hot = []
    for k in hot_k:  # hot spots alone: substring test off
        r = rates(replace(settings, hot_k=k, band_k=1e9))
        sweep_hot.append(
            {
                "hot_k": k,
                "false_alarm": r["No-Anomaly"].get("cell", 0.0),
                **{c: r[c].get("cell", 0.0) for c in HOT if c in r},
            }
        )
    sweep_band = []
    for k in band_k:  # substrings alone: hot-spot test off
        r = rates(replace(settings, band_k=k, hot_k=1e9))
        sweep_band.append(
            {
                "band_k": k,
                "false_alarm": r["No-Anomaly"].get("substring", 0.0),
                **{c: r[c].get("substring", 0.0) for c in DIODE if c in r},
            }
        )
    return {
        "images": len(idx),
        "settings": {"hot_k": settings.hot_k, "band_k": settings.band_k},
        "default": default,
        "sweep_hot": sweep_hot,
        "sweep_band": sweep_band,
    }
