"""Find thermal anomalies and name the module (and cell) they are on.

Per frame, every module of the rows the pass was planned for is projected into the image
with the logged camera pose and resampled into a rectified crop in module coordinates
(3 cm grid, top of the module up). Each crop is compared with itself and with its row
neighbours in the same frame:

* hot spots: connected pixels at least ``hot_k`` above the module's median;
* bypass-diode substring: one third of the module (two cell columns, full length) at least
  ``band_k`` warmer than the other two;
* offline module: the whole module at least ``offline_k`` warmer than its neighbours.

A hot spot is dropped as a likely sun reflection when the line of sight, mirrored in the
glass, points within ``reflection_guard_deg`` of the sun. Each module is seen in several
overlapping frames; a finding is kept only if most of those views agree, which also removes
reflections and noise that appear in one view only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pvmission.plant import Plant
from pvmission.sun import solar_position, sun_vector
from scipy import ndimage

from .camera import project
from .scene import CELLS_ACROSS, CELLS_ALONG, FRAME_M, SUBSTRINGS, Module, Pose, Scene


@dataclass(frozen=True)
class Settings:
    resolution_m: float = 0.03  # crop grid
    hot_k: float = 3.0  # hot spot: K above the module median
    hot_min_cells: float = 0.25  # minimum hot-spot area, in cells
    band_k: float = 2.0  # substring: K above the other two thirds
    band_spread: float = 1.5  # ... and this many times the difference between those two
    offline_k: float = 2.0  # offline: K above the row neighbours in the same frame
    reflection_guard_deg: float = 6.0
    min_views: int = 2
    min_view_fraction: float = 0.5
    string_min_modules: int = 3


@dataclass
class View:
    """What one frame shows of one module."""

    module_id: str
    frame: int
    kind: str  # "none", "cell", "substring", "offline"
    delta_t: float = 0.0
    cells: list[tuple[int, int]] = field(default_factory=list)
    blobs: int = 0
    substring: int = -1
    reflections: int = 0  # hot spots dropped as reflections
    median: float = float("nan")


@dataclass
class Finding:
    module_id: str
    kind: str  # hot_cell, hot_cells, substring, offline, string
    delta_t: float
    views_flagged: int
    views_seen: int
    cells: list[tuple[int, int]] = field(default_factory=list)
    substring: int = -1
    lat: float = float("nan")
    lon: float = float("nan")


def crop_grid(m: Module, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Module coordinates (ms, mt) of the crop pixels; row 0 is the top of the module."""
    nx, ny = int(round(m.width / res)), int(round(m.height / res))
    ms = (np.arange(nx) + 0.5) * m.width / nx
    mt = (np.arange(ny)[::-1] + 0.5) * m.height / ny
    return np.meshgrid(ms, mt)


def _hidden(P: np.ndarray, C: np.ndarray, rows) -> np.ndarray:
    """True where the line of sight from P to the camera C crosses one of ``rows``."""
    hit = np.zeros(P.shape[:-1], dtype=bool)
    v = C - P
    for r in rows:
        n = r.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((r.center - P) @ n) / (v @ n)
        X = P + t[..., None] * v - r.center
        inside = (np.abs(X @ r.along) <= r.length / 2) & (np.abs(X @ r.upslope) <= r.width / 2)
        hit |= (t > 1e-6) & (t < 1 - 1e-6) & inside
    return hit


def module_crop(image: np.ndarray, m: Module, pose: Pose, scene: Scene, res: float):
    """Rectified crop of module ``m`` (NaN where another row hides it), or None if the module
    is not fully inside the image or mostly hidden."""
    cam = scene.camera
    u, v, depth = project(m.corners(), pose, cam)
    if (
        (depth <= 0).any()
        or u.min() < 1
        or v.min() < 1
        or u.max() > cam.width_px - 2
        or v.max() > cam.height_px - 2
    ):
        return None
    ms, mt = crop_grid(m, res)
    P = m.point(ms, mt)
    uu, vv, _ = project(P, pose, cam)
    crop = ndimage.map_coordinates(image, [vv, uu], order=1, mode="nearest")
    near = [r for r in scene.rows if r is not m.row and abs(r.index - m.row.index) <= 2]
    hidden = _hidden(P, pose.position, near)
    if hidden.mean() > 0.5:
        return None
    if hidden.any():  # also drop a 2-pixel seam next to the hiding edge (lens blur)
        hidden = ndimage.binary_dilation(hidden, iterations=2)
        crop = np.where(hidden, np.nan, crop)
    return crop


def _cell_of(m: Module, ms: float, mt: float) -> tuple[int, int]:
    cw = (m.width - 2 * FRAME_M) / CELLS_ACROSS
    ch = (m.height - 2 * FRAME_M) / CELLS_ALONG
    return (
        int(np.clip((mt - FRAME_M) // ch, 0, CELLS_ALONG - 1)),
        int(np.clip((ms - FRAME_M) // cw, 0, CELLS_ACROSS - 1)),
    )


def analyse_crop(
    crop: np.ndarray, m: Module, st: Settings, res: float, relative: bool = False
) -> dict:
    """Hot spots and substring pattern within one module crop (NaN pixels are ignored).

    ``relative=True`` is for images without temperatures (8-bit, contrast-stretched): the
    thresholds are then in robust standard deviations instead of kelvin.
    """
    ms, mt = crop_grid(m, res)
    interior = (ms > FRAME_M + res) & (ms < m.width - FRAME_M - res)
    interior &= (mt > FRAME_M + res) & (mt < m.height - FRAME_M - res)
    interior &= np.isfinite(crop)
    vals = crop[interior]
    med = float(np.median(vals))
    sigma = max(1.4826 * float(np.median(np.abs(vals - med))), 1e-6)
    scale = sigma if relative else 1.0
    cw = (m.width - 2 * FRAME_M) / CELLS_ACROSS
    ch = (m.height - 2 * FRAME_M) / CELLS_ALONG
    band = np.clip(((ms - FRAME_M) // (2 * cw)).astype(int), 0, SUBSTRINGS - 1)
    band_med = np.array(
        [
            np.median(crop[interior & (band == b)]) if (interior & (band == b)).any() else med
            for b in range(SUBSTRINGS)
        ]
    )
    hot_band = int(np.argmax(band_med))
    others = np.delete(band_med, hot_band)
    band_dt = float(band_med[hot_band] - np.median(others))
    band_scale = 1.0
    if relative:  # noise of the two other thirds: a warm third would inflate the whole-crop MAD
        rest = crop[interior & (band != hot_band)]
        band_scale = (
            max(1.4826 * float(np.median(np.abs(rest - np.median(rest)))), 1e-6)
            if rest.size
            else sigma
        )
    is_substring = band_dt >= st.band_k * band_scale and band_dt >= st.band_spread * (
        others.max() - others.min()
    )
    ref = float(np.median(others)) if is_substring else med
    work = np.where(interior, crop, -np.inf)
    hot = interior & (work - ref >= st.hot_k * scale)
    if is_substring:  # the warm band itself is not a hot spot
        hot &= band != hot_band
        hot |= interior & (band == hot_band) & (work - band_med[hot_band] >= st.hot_k * scale)
    labels, n = ndimage.label(hot, structure=np.ones((3, 3)))
    cell_px = cw * ch / res**2
    size = max(3, int(round(0.8 * min(cw, ch) / res)))
    local_max = (work == ndimage.maximum_filter(work, size=size)) & hot
    blobs = []
    for i in range(1, n + 1):
        px = labels == i
        if px.sum() < st.hot_min_cells * cell_px:
            continue
        w = work[px] - ref
        peak = float(w.max())
        # separate hot cells inside one warm region: local maxima of at least half the peak,
        # at most one per cell
        pk = local_max & px & (work - ref >= 0.5 * peak)
        cells = sorted({_cell_of(m, a, b) for a, b in zip(ms[pk], mt[pk], strict=True)})
        cx, cy = float(np.average(ms[px], weights=w)), float(np.average(mt[px], weights=w))
        blobs.append(
            {
                "ms": cx,
                "mt": cy,
                "peak": peak,
                "area_px": int(px.sum()),
                "cell": _cell_of(m, cx, cy),
                "cells": cells or [_cell_of(m, cx, cy)],
            }
        )
    return {
        "median": med,
        "sigma": sigma,
        "substring": hot_band if is_substring else -1,
        "band_dt": band_dt,
        "blobs": blobs,
    }


class Mapper:
    def __init__(self, scene: Scene, settings: Settings = Settings()):
        self.scene, self.st = scene, settings
        self.plant = Plant(scene.name, scene.frame, scene.rows, scene.timezone)
        m = scene.extras.get("mission", {})
        self.line_rows = {ln["index"]: [r - 1 for r in ln["rows"]] for ln in m.get("lines", [])}

    def _sun(self, time: pd.Timestamp) -> np.ndarray:
        sp = solar_position(self.plant, pd.DatetimeIndex([time]))
        return sun_vector(sp.apparent_elevation.iloc[0], sp.azimuth.iloc[0])

    def frame_views(
        self, index: int, image: np.ndarray, pose: Pose, relative: bool = False
    ) -> list[View]:
        st, res = self.st, self.st.resolution_m
        rows = self.line_rows.get(pose.line, range(len(self.scene.rows)))
        sun = self._sun(pose.time) if pose.time is not None else None
        views: list[View] = []
        for ri in rows:
            row = self.scene.rows[ri]
            row_views, medians = [], []
            for c in range(1, row.modules_along + 1):
                for lvl in range(1, row.modules_across + 1):
                    m = Module(row, c, lvl)
                    crop = module_crop(image, m, pose, self.scene, res)
                    if crop is None:
                        continue
                    a = analyse_crop(crop, m, st, res, relative)
                    blobs = a["blobs"]
                    n_refl = 0
                    if sun is not None and blobs:
                        keep = []
                        for b in blobs:
                            P = m.point(b["ms"], b["mt"])
                            d = (P - pose.position) / np.linalg.norm(P - pose.position)
                            n = row.normal
                            mirrored = d - 2 * (d @ n) * n
                            glare = np.degrees(np.arccos(np.clip(mirrored @ sun, -1, 1)))
                            if glare < st.reflection_guard_deg:
                                n_refl += 1
                            else:
                                keep.append(b)
                        blobs = keep
                    v = View(m.id, index, "none", median=a["median"], reflections=n_refl)
                    if a["substring"] >= 0:
                        v.kind, v.delta_t, v.substring = "substring", a["band_dt"], a["substring"]
                    elif blobs:
                        v.kind = "cell"
                        v.delta_t = max(b["peak"] for b in blobs)
                        v.blobs = len(blobs)
                        v.cells = sorted({c for b in blobs for c in b["cells"]})
                    row_views.append(v)
                    medians.append(a["median"])
            if not row_views:
                continue
            ref = float(np.median(medians))
            scale = 1.0
            if relative:
                scale = max(1.4826 * float(np.median(np.abs(np.array(medians) - ref))), 1e-6)
            for v in row_views:  # whole module warmer than its neighbours in this frame
                if v.kind == "none" and v.median - ref >= st.offline_k * scale:
                    v.kind, v.delta_t = "offline", v.median - ref
            views += row_views
        return views

    def fuse(self, views: list[View]) -> list[Finding]:
        st = self.st
        by_module: dict[str, list[View]] = defaultdict(list)
        for v in views:
            by_module[v.module_id].append(v)
        findings = []
        for mid, vs in by_module.items():
            flagged = [v for v in vs if v.kind != "none"]
            if not flagged:
                continue
            kind, n = Counter(v.kind for v in flagged).most_common(1)[0]
            need = (
                1
                if len(vs) == 1
                else max(st.min_views, int(np.ceil(st.min_view_fraction * len(vs))))
            )
            if n < need:
                continue
            agree = [v for v in flagged if v.kind == kind]
            dt = float(np.median([v.delta_t for v in agree]))
            f = Finding(mid, kind, round(dt, 2), n, len(vs))
            if kind == "cell":
                cells = Counter(c for v in agree for c in v.cells)
                core = sorted(c for c, k in cells.items() if k >= max(1, (n + 1) // 2))
                f.cells = core or [cells.most_common(1)[0][0]]
                per_view = int(np.median([len(v.cells) for v in agree]))
                f.kind = "hot_cell" if per_view <= 1 else "hot_cells"
            elif kind == "substring":
                f.substring = Counter(v.substring for v in agree).most_common(1)[0][0]
            m = self.scene.module(mid)
            lat, lon = self.scene.frame.to_geodetic(*m.center[:2])
            f.lat, f.lon = float(lat), float(lon)
            findings.append(f)
        return self._strings(findings)

    def _strings(self, findings: list[Finding]) -> list[Finding]:
        """Runs of neighbouring offline modules (same row and level) are one offline string."""
        off = defaultdict(list)
        for f in findings:
            if f.kind == "offline":
                r, c, lvl = f.module_id.split("-")
                off[(r, lvl)].append((int(c[1:]), f))
        for items in off.values():
            items.sort(key=lambda x: x[0])
            run = [items[0]] if items else []
            for prev, cur in zip(items, items[1:], strict=False):
                if cur[0] == prev[0] + 1:
                    run.append(cur)
                    continue
                self._mark(run)
                run = [cur]
            self._mark(run)
        return sorted(findings, key=lambda f: f.module_id)

    def _mark(self, run) -> None:
        if len(run) >= self.st.string_min_modules:
            for _, f in run:
                f.kind = "string"

    def run(
        self, frames: list[tuple[np.ndarray, Pose]], relative: bool = False
    ) -> tuple[list[Finding], list[View]]:
        views = []
        for i, (img, pose) in enumerate(frames):
            views += self.frame_views(i, img, pose, relative)
        return self.fuse(views), views


def load_survey(folder: Path) -> list[tuple[np.ndarray, Pose]]:
    """Frames listed in frames.json with their logged poses, as deg C.

    Images are TIFF/PNG or .npy, in the ``units`` frames.json gives: "centikelvin" (16-bit
    radiometric, as thermal cameras store raw data), "kelvin" or "degC" (float).
    """
    folder = Path(folder)
    index = json.loads((folder / "frames.json").read_text())
    units = index.get("units", "degC")
    offset, scale = {"centikelvin": (-273.15, 0.01), "kelvin": (-273.15, 1.0), "degC": (0.0, 1.0)}[
        units
    ]
    out = []
    for f in index["frames"]:
        path = folder / f["file"]
        raw = np.load(path) if path.suffix == ".npy" else np.asarray(Image.open(path))
        img = (raw.astype(np.float32) * scale + offset).astype(np.float32)
        pose = Pose(
            np.array(f["enu"], float),
            f["yaw_deg"],
            f["gimbal_pitch_deg"],
            pd.Timestamp(f["time"]),
            f["line"],
            f["seq"],
        )
        out.append((img, pose))
    return out


def findings_table(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in findings])
