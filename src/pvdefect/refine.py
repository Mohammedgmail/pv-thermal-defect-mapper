"""Correct logged camera poses with the plant itself as the calibration target.

Small heading and gimbal errors move the image by a few decimetres at the module, enough to
put a hot cell on the wrong cell or a defect near a module edge on the wrong module. A
non-RTK GNSS position is off by a metre or more, about one module width.

Each frame is resampled onto the plane of the rows its pass covers (an orthophoto in row
coordinates s, along the row, and t, up the slope). There the expected structure is known
exactly: cool aluminium frames between modules (every module width along s), at the level
boundary and at the table edges, with ground beyond. Per frame:

* ``t`` offset: match the observed profile across the table to the expected one (unique);
* ``s`` offset modulo one module width: match the periodic frame lines along the row.

Along the row the frame lines repeat, so the absolute ``s`` offset is fixed per pass: the
first and last frames of a pass see the end of the row, and consecutive frames are chained by
correlating the modules' own thermal texture where the frames overlap. The correction is
applied as a camera shift parallel to the module plane, which absorbs most of the heading,
pitch and position error.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .camera import project
from .mapper import _hidden
from .scene import Pose, Scene

RES = 0.03  # m


def _nanmedian(x: np.ndarray, axis: int | None) -> np.ndarray:
    """nanmedian without the warning for all-NaN columns (parts of the ortho outside the image)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def ortho(
    image: np.ndarray,
    pose: Pose,
    scene: Scene,
    row,
    s: np.ndarray,
    t: np.ndarray,
    mask_hidden: bool = True,
) -> np.ndarray:
    """Image resampled onto row coordinates (len(t), len(s)); NaN outside the image or (with
    ``mask_hidden``) where another row hides the plane as seen from ``pose``.

    The hidden mask is only right when the pose is: with a pose error of a metre it lands on
    the strip of ground below the table, which is the one feature that fixes the across-row
    offset, so the first, coarse pass works without it.
    """
    S, T = np.meshgrid(s, t)
    P = row.point(S, T)
    u, v, depth = project(P, pose, scene.camera)
    cam = scene.camera
    out = ndimage.map_coordinates(image, [v, u], order=1, mode="constant", cval=np.nan)
    bad = (depth <= 0) | (u < 0) | (v < 0) | (u > cam.width_px - 1) | (v > cam.height_px - 1)
    if mask_hidden:
        near = [r for r in scene.rows if r is not row and abs(r.index - row.index) <= 2]
        bad |= _hidden(P, pose.position, near)
    return np.where(bad, np.nan, out)


def _highpass(o: np.ndarray, size_m: float = 0.3) -> np.ndarray:
    filled = np.where(np.isfinite(o), o, _nanmedian(o, None))
    return o - ndimage.median_filter(filled, size=max(3, int(size_m / RES)))


def _line_score(
    profile: np.ndarray, coords: np.ndarray, lines: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    """Mean of ``profile`` at the frame-line positions for each candidate shift (lower = cooler)."""
    good = np.isfinite(profile)
    out = np.full(len(shifts), np.inf)
    for i, d in enumerate(shifts):
        pos = lines + d
        pos = pos[(pos > coords[0]) & (pos < coords[-1])]
        if len(pos) == 0:
            continue
        vals = np.interp(pos, coords[good], profile[good]) if good.sum() > 2 else np.array([np.inf])
        out[i] = np.mean(vals)
    return out


def _lower_edge(o: np.ndarray, t: np.ndarray, row, search: float) -> float | None:
    """Coarse across-row offset from the lower table edge: warm modules above, cooler ground
    below (the camera looks at the module faces, so the ground in front is in view)."""
    prof = _nanmedian(o, axis=1)
    win = int(0.5 / RES)
    best = None
    for i in np.flatnonzero(np.abs(t + row.width / 2) <= search):
        if i < win or i + win > len(prof):
            continue
        below, above = prof[i - win : i], prof[i : i + win]
        if not (np.isfinite(below).all() and np.isfinite(above).all()):
            continue
        step = float(above.mean() - below.mean())
        if step > 5.0 and (best is None or step > best[0]):
            best = (step, float(t[i] + row.width / 2))
    return None if best is None else best[1]


def _t_offset(
    o: np.ndarray, t: np.ndarray, row, search: float, prior_k_per_m: float = 0.5
) -> float:
    """Across-row offset: coarse from the lower table edge, fine from the cool frame lines at
    both table edges and between levels (frame lines should be cool, module interiors not)."""
    coarse = _lower_edge(o, t, row, search)
    centre = 0.0 if coarse is None else coarse
    window = 0.3 if coarse is not None else min(search, 0.45 * row.width / row.modules_across)
    prof = _nanmedian(_highpass(o, 0.3), axis=1)
    h = row.width / row.modules_across
    lines = -row.width / 2 + np.arange(row.modules_across + 1) * h
    inner = np.concatenate(
        [
            np.arange(-row.width / 2 + 0.15, -row.width / 2 + h - 0.15, RES * 2) + j * h
            for j in range(row.modules_across)
        ]
    )
    shifts = np.arange(centre - window, centre + window + 1e-9, RES / 3)
    score = _line_score(prof, t, lines, shifts) - _line_score(prof, t, inner, shifts)
    return float(shifts[int(np.argmin(score + prior_k_per_m * np.abs(shifts - centre)))])


def _s_modulo(o: np.ndarray, s: np.ndarray, row) -> tuple[float, float]:
    """Along-row offset modulo one module width, and how clear the frame lines are."""
    w = row.length / row.modules_along
    prof = _nanmedian(_highpass(o), axis=0)
    lines = -row.length / 2 + np.arange(row.modules_along + 1) * w
    shifts = np.arange(-w / 2, w / 2, RES / 3)
    score = _line_score(prof, s, lines, shifts)
    i = int(np.argmin(score))
    contrast = float(_nanmedian(prof, None) - score[i])
    return float(shifts[i]), contrast


def _row_end(o: np.ndarray, s: np.ndarray, row, search: float) -> float | None:
    """Offset of a visible row end, else None.

    A row end is where the mean temperature over a metre of modules drops to ground level
    over the next metre (a thin cool frame line barely moves a one-metre mean).
    """
    prof = _nanmedian(o, axis=0)
    good = np.isfinite(prof)
    win = int(1.0 / RES)
    best = None
    for end, modules_below in ((row.length / 2, True), (-row.length / 2, False)):
        for i in np.flatnonzero(good & (np.abs(s - end) <= search)):
            if i < win or i + win > len(prof):
                continue
            below, above = prof[i - win : i], prof[i : i + win]
            if not (np.isfinite(below).all() and np.isfinite(above).all()):
                continue
            inner, outer = (below, above) if modules_below else (above, below)
            contrast = float(inner.mean() - outer.mean())
            if contrast > 5.0 and (best is None or contrast > best[0]):
                best = (contrast, float(s[i] - end))
    return None if best is None else best[1]


def _s_scale(
    o: np.ndarray,
    s: np.ndarray,
    row,
    s0: float,
    lo: float = 0.85,
    hi: float = 1.15,
    step: float = 0.0025,
) -> float:
    """Apparent scale of the frame-line spacing along the row, about ``s0``.

    A camera that is really closer to or further from the module plane than logged sees the
    plane scaled about its foot point: frame lines appear every ``k * w`` instead of every
    ``w``, with ``k`` the ratio of logged to true distance. Over a 20 m footprint a 2 m height
    error moves the outer modules by most of a module width, so this is fixed first.
    """
    w = row.length / row.modules_along
    prof = _nanmedian(_highpass(o), axis=0)
    good = np.isfinite(prof)
    if good.sum() < 4 * w / RES:
        return 1.0
    sg, pg = s[good], prof[good]
    best_k, best = 1.0, np.inf
    for k in np.arange(lo, hi + 1e-9, step):
        period = k * w
        m = np.arange(np.floor((sg[0] - s0) / period) - 1, np.ceil((sg[-1] - s0) / period) + 2)
        pos = s0 + m[:, None] * period + np.arange(0, period, RES / 2)[None, :]
        inside = (pos > sg[0]) & (pos < sg[-1])
        score = np.where(inside, np.interp(pos, sg, pg), 0).sum(0) / np.maximum(inside.sum(0), 1)
        if score.min() < best:
            best, best_k = float(score.min()), float(k)
    return best_k


def _t_pass(frames, ks, poses, scene, row, s_of, t, search_t) -> tuple[dict[int, float], float]:
    """Across-row offset of every frame of a pass, with the pass median for outliers, and the
    fraction of frames in which the lower table edge was found.

    The across-row error is mostly a slowly varying bias: a frame that disagrees with the pass
    by more than half a metre has locked onto the wrong frame line.
    """
    et, edges = {}, 0
    for k in ks:
        o = ortho(frames[k][0], poses[k], scene, row, s_of[k], t, mask_hidden=False)
        et[k] = _t_offset(o, t, row, search_t)
        edges += _lower_edge(o, t, row, search_t) is not None
    med = float(np.median(list(et.values())))
    return {k: (med if abs(v - med) > 0.5 else v) for k, v in et.items()}, edges / len(ks)


@dataclass
class PassReport:
    """What the refinement did on one pass."""

    line: int
    frames: int
    applied: bool  # False: the lower table edge was not found, the logged poses are kept
    edge_found: float  # fraction of frames in which the lower table edge was found
    scale: float  # logged / true distance to the module plane
    row_end_frames: int  # frames that saw a row end (absolute along-row anchor)
    shift_m: float  # median correction at the module plane

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def refine_poses(
    scene: Scene,
    frames: list[tuple[np.ndarray, Pose]],
    search_t: float = 2.5,
    search_s: float = 4.0,
) -> tuple[list[Pose], list[PassReport]]:
    """Corrected poses for ``frames`` (image, logged pose), in the same order, and a report
    per pass. A pass whose lower table edge is not found within ``search_t`` of where the log
    puts it (a position error of more than about 2.5 m) keeps its logged poses."""
    lines = scene.extras.get("mission", {}).get("lines", [])
    line_rows = {ln["index"]: [r - 1 for r in ln["rows"]] for ln in lines}
    by_line = defaultdict(list)
    for k, (_, p) in enumerate(frames):
        by_line[p.line].append(k)
    out = [p for _, p in frames]
    reports = []
    for line, ks in sorted(by_line.items()):
        row = scene.rows[line_rows.get(line, [0])[0]]
        a, up = row.along, row.upslope
        w = row.length / row.modules_along
        t = np.arange(-row.width / 2 - search_t - 0.8, row.width / 2 + search_t + 0.8, RES)
        s_all = np.arange(-row.length / 2 - search_s - 12, row.length / 2 + search_s + 12, RES)
        ks = sorted(ks, key=lambda k: (frames[k][1].position - row.center) @ a)
        s_of = {k: s_all[np.abs(s_all - (frames[k][1].position - row.center) @ a) < 10] for k in ks}
        table = np.abs(t) < row.width / 2 - 0.1  # along-row profiles: modules only
        poses = {k: frames[k][1] for k in ks}
        # 1. across-row offset, 2. distance to the module plane (pass median of the scale),
        # 3. across-row offset again (a distance change moves an oblique view across the row)
        et, edge_found = _t_pass(frames, ks, poses, scene, row, s_of, t, search_t)
        if edge_found < 0.5:
            reports.append(PassReport(line, len(ks), False, edge_found, 1.0, 0, 0.0))
            continue
        scale = []
        for k in ks:
            p = poses[k].moved(-et[k] * up)
            o = ortho(frames[k][0], p, scene, row, s_of[k], t)[table]
            scale.append(_s_scale(o, s_of[k], row, (p.position - row.center) @ a))
        k_med = float(np.median(scale))
        for k in ks:
            d = (poses[k].position - row.center) @ row.normal
            poses[k] = poses[k].moved((d / k_med - d) * row.normal)
        et, _ = _t_pass(frames, ks, poses, scene, row, s_of, t, search_t)
        # 4. along-row offset modulo one module, and the row ends where visible
        emod, ends = {}, {}
        for k in ks:
            o = ortho(frames[k][0], poses[k].moved(-et[k] * up), scene, row, s_of[k], t)[table]
            emod[k], _ = _s_modulo(o, s_of[k], row)
            ends[k] = _row_end(o, s_of[k], row, search_s)
        # absolute offsets e = emod + n * w. The row ends fix e in the first and last frames;
        # in between, the error of a pass changes slowly (a GNSS bias drifts over minutes, not
        # seconds), so e is interpolated between the two ends and each frame takes the n that
        # brings it closest. Without a visible end, trust the log to half a module.
        pos = {k: j for j, k in enumerate(ks)}
        anchors = [(pos[k], ends[k]) for k in ks if ends[k] is not None]
        e_pred = {k: 0.0 for k in ks}
        if anchors:
            x, y = np.array(anchors).T
            first, last = x < len(ks) / 2, x >= len(ks) / 2
            if first.any() and last.any():
                x0, y0 = np.median(x[first]), np.median(y[first])
                x1, y1 = np.median(x[last]), np.median(y[last])
                e_pred = {k: float(y0 + (pos[k] - x0) * (y1 - y0) / (x1 - x0)) for k in ks}
            else:
                e_pred = {k: float(np.median(y)) for k in ks}
        n = {k: int(np.round((e_pred[k] - emod[k]) / w)) for k in ks}
        shifts = []
        for k in ks:
            e = emod[k] + n[k] * w
            out[k] = poses[k].moved(-et[k] * up - e * a)
            shifts.append(np.hypot(et[k], e))
        reports.append(
            PassReport(
                line, len(ks), True, edge_found, k_med, len(anchors), float(np.median(shifts))
            )
        )
    return out, reports
