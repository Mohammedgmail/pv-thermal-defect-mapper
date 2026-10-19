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

Along the row the frame lines repeat, so the whole module offset is decided per pass: the
first and last frames of a pass see the end of the row (a drop to ground temperature, or
where the frame lines stop when sunlit ground is as warm as the modules), and consecutive
frames are linked by correlating the modules' own thermal texture where they overlap. One
dynamic programme over the frames of the pass weighs the two, so a quick GNSS jump between
frames or a wrong row-end reading does not move a frame onto the neighbouring module. The
correction is applied as a camera shift parallel to the module plane, which absorbs most of
the heading, pitch and position error.

Thresholds are relative to each frame's own contrast where they can be, so the same code
runs on radiometric frames in kelvin and on raw counts with an unknown scale.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from .camera import focal_px
from .mapper import _hidden
from .scene import Pose, Scene

RES = 0.03  # m
EDGE_K = 5.0  # temperature step (K) that always counts as a table edge (and as a row end)
EDGE_FRACTION = 0.35  # a table edge also counts above this share of the frame's contrast


def _nanmedian(x: np.ndarray, axis: int | None) -> np.ndarray:
    """Median ignoring NaN (NaN where a slice is all NaN: parts of the ortho outside the
    image). Same values as ``np.nanmedian``, but sorts the whole array at once instead of
    looping over slices, which is what made the profiles slow."""
    x = np.asarray(x, dtype=float)
    if axis is None:
        x = x[~np.isnan(x)]
        return np.median(x) if x.size else np.float64(np.nan)
    n = (~np.isnan(x)).sum(axis=axis)
    xs = np.moveaxis(np.sort(x, axis=axis), axis, -1)  # NaN sort to the end
    lo = np.take_along_axis(xs, np.maximum((n - 1) // 2, 0)[..., None], -1)[..., 0]
    hi = np.take_along_axis(xs, np.maximum(n // 2, 0)[..., None], -1)[..., 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.where(n > 0, (lo + hi) / 2, np.nan)


def _plane_uv(pose: Pose, cam, row, s: np.ndarray, t: np.ndarray):
    """Pixel coordinates and depth of the row-plane grid (len(t), len(s)). The camera
    coordinates of a plane point are affine in (s, t), so no 3-D grid is built."""
    x, y, z = pose.axes()
    c = row.center - pose.position
    a, w = row.along, row.upslope

    def lin(e):
        return (c @ e) + (a @ e) * s[None, :] + (w @ e) * t[:, None]

    depth = lin(z)
    f = focal_px(cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (cam.width_px - 1) / 2 + f * lin(x) / depth
        v = (cam.height_px - 1) / 2 + f * lin(y) / depth
    return u, v, depth


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at (u, v); the caller masks points outside the image."""
    h, w = image.shape
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)
    x0 = np.clip(np.floor(u), 0, w - 2)
    y0 = np.clip(np.floor(v), 0, h - 2)
    fx, fy = u - x0, v - y0
    i = y0.astype(np.intp) * w + x0.astype(np.intp)
    flat = image.ravel()
    top = flat[i] * (1 - fx) + flat[i + 1] * fx
    bottom = flat[i + w] * (1 - fx) + flat[i + w + 1] * fx
    return top * (1 - fy) + bottom * fy


def _hidden_on_plane(row, s: np.ndarray, t: np.ndarray, C: np.ndarray, rows) -> np.ndarray:
    """``mapper._hidden`` for the grid ``row.point(s, t)``. For a row parallel to ``row``,
    the line of sight from any point of the plane meets it at the same fraction of the way to
    the camera, so the test separates into one mask along s and one across t."""
    hit = np.zeros((len(t), len(s)), dtype=bool)
    n, a, w = row.normal, row.along, row.upslope
    pn, cn = row.center @ n, C @ n
    pa, pw = row.center @ a + s, row.center @ w + t
    for r in rows:
        if not (np.allclose(r.normal, n, atol=1e-12) and np.allclose(r.along, a, atol=1e-12)):
            S, T = np.meshgrid(s, t)
            hit |= _hidden(row.point(S, T), C, [r])
            continue
        if cn == pn:
            continue
        lam = (r.center @ n - pn) / (cn - pn)
        if not 1e-6 < lam < 1 - 1e-6:
            continue
        xa = pa + lam * (C @ a - pa) - r.center @ a
        xw = pw + lam * (C @ w - pw) - r.center @ w
        hit |= (np.abs(xw) <= r.width / 2)[:, None] & (np.abs(xa) <= r.length / 2)[None, :]
    return hit


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
    cam = scene.camera
    u, v, depth = _plane_uv(pose, cam, row, s, t)
    out = _bilinear(image, u, v)
    bad = ~(depth > 0) | ~(u >= 0) | ~(v >= 0) | (u > cam.width_px - 1) | (v > cam.height_px - 1)
    if mask_hidden:
        near = [r for r in scene.rows if r is not row and abs(r.index - row.index) <= 2]
        bad |= _hidden_on_plane(row, s, t, pose.position, near)
    return np.where(bad, np.nan, out)


def _highpass(o: np.ndarray, size_m: float = 0.3) -> np.ndarray:
    """Image minus a local median background (separable: along each axis in turn, which is
    much faster than a square window and as good for removing smooth trends)."""
    filled = np.where(np.isfinite(o), o, _nanmedian(o, None))
    k = max(3, int(size_m / RES))
    background = ndimage.median_filter(ndimage.median_filter(filled, size=(1, k)), size=(k, 1))
    return o - background


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


def _edge_threshold(prof: np.ndarray) -> float:
    """Smallest step accepted as an edge: EDGE_K, or a share of the profile's own contrast
    when that is lower (a cool morning, or raw counts with an unknown scale)."""
    good = prof[np.isfinite(prof)]
    if good.size < 10:
        return EDGE_K
    lo, hi = np.percentile(good, [5, 95])
    return float(min(EDGE_K, EDGE_FRACTION * (hi - lo)))


def _steps(prof: np.ndarray, win: int, gap: int) -> np.ndarray:
    """Mean of ``win`` samples above minus mean of ``win`` samples below each index, leaving
    ``gap`` samples free on either side (NaN where a window is not fully covered)."""
    out = np.full(len(prof), np.nan)
    c = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(prof), prof, 0.0))])
    n = np.concatenate([[0], np.cumsum(np.isfinite(prof))])
    i = np.arange(win + gap, len(prof) - win - gap + 1)
    lo0, lo1, hi0, hi1 = i - gap - win, i - gap, i + gap, i + gap + win
    full = (n[lo1] - n[lo0] == win) & (n[hi1] - n[hi0] == win)
    step = (c[hi1] - c[hi0] - c[lo1] + c[lo0]) / win
    out[i[full]] = step[full]
    return out


def _lower_edge(
    o: np.ndarray, t: np.ndarray, row, search: float, gap: float = 0.15
) -> float | None:
    """Coarse across-row offset from the lower table edge: warm modules above, cooler ground
    below (the camera looks at the module faces, so the ground in front is in view).

    Found where the mean over half a metre above exceeds the mean over half a metre below,
    leaving ``gap`` free on either side, where the frame, dirt at the lower edge and lens
    blur can smear the step; then placed at the sharpest step within that gap."""
    prof = _nanmedian(o, axis=1)
    threshold = _edge_threshold(prof)
    win, g = int(0.5 / RES), int(round(gap / RES))
    near = np.abs(t + row.width / 2) <= search
    wide = np.where(near, _steps(prof, win, g), np.nan)
    if not np.isfinite(wide).any() or np.nanmax(wide) <= threshold:
        return None
    i = int(np.nanargmax(wide))
    sharp = _steps(prof, win, 0)[max(i - g, 0) : i + g + 1]
    if np.isfinite(sharp).any():
        i = max(i - g, 0) + int(np.nanargmax(sharp))
    return float(t[i] + row.width / 2)


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
    threshold = EDGE_K  # absolute: warm ground in the frame must not pass as a row end
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
            if contrast > threshold and (best is None or contrast > best[0]):
                best = (contrast, float(s[i] - end))
    return None if best is None else best[1]


def _dips(prof: np.ndarray, s: np.ndarray, pos: np.ndarray, snap: float = 0.09) -> np.ndarray:
    """How much cooler the profile is on a thin line near each position than the cooler of
    its two sides 15-35 cm away, so that only a line cooler than both sides counts (a step
    from modules down to cool ground does not); NaN where the profile does not cover it.
    The line is looked for within ``snap`` of the position: the lattice drifts by a few
    centimetres towards the image edges."""
    out = np.full(len(pos), np.nan)
    for i, p0 in enumerate(pos):
        best = np.nan
        for p in p0 + np.arange(-snap, snap + 1e-9, RES):
            d = s - p
            on = prof[np.abs(d) <= 0.045]
            left, right = prof[(d >= -0.35) & (d <= -0.15)], prof[(d >= 0.15) & (d <= 0.35)]
            if all(x.size and np.isfinite(x).all() for x in (on, left, right)):
                v = on.mean() - min(left.mean(), right.mean())
                best = v if not best <= v else best
        out[i] = best
    return out


def _row_end_pattern(o: np.ndarray, s: np.ndarray, row, e_mod: float, search: float):
    """Offset of a visible row end from where the frame lines stop, else None.

    Sunlit ground beyond a row can be as warm as the modules, and then there is no drop in
    temperature at the end. The cool frame lines between modules stop there too. They lie on
    the lattice of the model's lines shifted by ``e_mod``: inside the row every lattice point
    is a clear dip (a line cooler than both its sides), beyond the end none is. The outer
    frame at the row end has ground on one side and is not a clear dip. Scanning in from
    outside, the end is the first lattice point that is not a dip, has flat ground beyond it
    and two dips (the first at least half as deep as a typical one) inside it.

    Only ends in the middle of the image count: towards the image edge a small scale or
    pitch error moves the lattice off the lines, which then fade gradually instead of
    stopping, and look like an end a module or two too early.
    """
    w = row.length / row.modules_along
    prof = _nanmedian(o, axis=0)  # not high-passed: that turns a step to cool ground into a dip
    seen = s[np.isfinite(prof)]
    if seen.size < 10:
        return None
    mid, half = (seen[0] + seen[-1]) / 2, (seen[-1] - seen[0]) / 2
    n = int(np.ceil(search / w)) + 4
    i = np.arange(-n, n + 1)
    found = []
    for end, inward in ((-row.length / 2, 1), (row.length / 2, -1)):
        d = _dips(prof, s, end + e_mod + i * w)
        if np.isfinite(d).sum() < 6:
            continue
        line_k = 0.4 * min(float(np.nanpercentile(d, 20)), -0.5)  # a clear frame line
        line = d < line_k
        if line.sum() < 3:
            continue
        deep = 0.5 * float(np.median(d[line]))
        flat = np.abs(d) < 0.5 * abs(line_k)
        unseen = np.isnan(d)
        order = range(3, len(i) - 3) if inward > 0 else range(len(i) - 4, 2, -1)
        for j in order:  # from outside in
            beyond = (j - inward, j - 2 * inward)
            if (
                d[j + inward] < deep
                and line[j + 2 * inward]
                and not line[j]
                and all(flat[b] or unseen[b] for b in beyond)
                and any(flat[b] for b in beyond)
            ):
                e = float(e_mod + i[j] * w)
                if abs(e) <= search and abs(end + e - mid) <= 0.6 * half:
                    found.append(e)
                break
    return found[0] if len(found) == 1 else None


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
    search_m: float = 2.5  # how far across the row the table edge was looked for

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


ALL_STEPS = ("t", "scale", "s", "anchors", "chain")


def _t_fine(o: np.ndarray, t: np.ndarray, row, window: float = 0.3) -> float:
    """Across-row offset from the frame lines only, within ``window`` of zero (for a pose that
    is already roughly right)."""
    prof = _nanmedian(_highpass(o, 0.3), axis=1)
    h = row.width / row.modules_across
    lines = -row.width / 2 + np.arange(row.modules_across + 1) * h
    inner = np.concatenate(
        [
            np.arange(-row.width / 2 + 0.15, -row.width / 2 + h - 0.15, RES * 2) + j * h
            for j in range(row.modules_across)
        ]
    )
    shifts = np.arange(-window, window + 1e-9, RES / 3)
    score = _line_score(prof, t, lines, shifts) - _line_score(prof, t, inner, shifts)
    return float(shifts[int(np.argmin(score + 0.5 * np.abs(shifts)))])


# along-row alignment: costs of a path of module offsets through a pass
TEXTURE_COST = 5.0  # per unit of texture correlation below the best step
TEXTURE_MIN = (0.3, 0.1)  # a texture match counts if it correlates this well, this clearly
ANCHOR_COST = 2.0  # a frame disagrees with its own row-end anchor
PRIOR_COST = 0.05  # per module of offset from the anchors interpolated along the pass
SMOOTH_COST = 0.3  # per module of change between consecutive frames


def _texture(
    hp: np.ndarray, s: np.ndarray, t: np.ndarray, e_mod: float, w: float, row
) -> np.ndarray:
    """The modules' own thermal pattern: the high-passed ortho inside the modules, minus its
    average over all modules at the same position within a module. Strips along all frame
    lines and the image border are left out: how deep a three-centimetre line looks depends
    on where it falls on the pixel grid, a pattern that moves with the camera, not with the
    modules, and the background filter is biased next to the border."""
    x = (s - e_mod + row.length / 2) % w
    phase = np.floor(x / RES).astype(int)
    out = np.full_like(hp, np.nan)
    for b in np.unique(phase):
        cols = phase == b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:, cols] = hp[:, cols] - np.nanmean(hp[:, cols], axis=1, keepdims=True)
    out[:, (x < 0.1) | (x > w - 0.1)] = np.nan
    h = row.width / row.modules_across
    y = (t + row.width / 2) % h
    out[(y < 0.1) | (y > h - 0.1), :] = np.nan
    border = ndimage.binary_dilation(~np.isfinite(hp), iterations=int(0.3 / RES))
    out[border] = np.nan
    return out


def _texture_match(tex, s_of, emod, k, k2, w, max_steps: int = 2) -> np.ndarray:
    """Correlation of the module texture of frames k and k2 where they overlap, for each
    change of the integer module offset from -max_steps to +max_steps (NaN: overlap under
    two metres). Only the right step lines the texture up."""
    a, b, sa, sb = tex[k], tex[k2], s_of[k], s_of[k2]
    out = np.full(2 * max_steps + 1, np.nan)
    for i, dn in enumerate(range(-max_steps, max_steps + 1)):
        j = np.round((sa + emod[k2] - emod[k] + dn * w - sb[0]) / RES).astype(int)
        ok = (j >= 0) & (j < len(sb))
        if ok.sum() * RES < 2.0:
            continue
        x, y = a[:, ok].ravel(), b[:, j[ok]].ravel()
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 500:
            continue
        x, y = x[good] - x[good].mean(), y[good] - y[good].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        out[i] = float((x * y).sum() / den) if den > 0 else np.nan
    return out


def _module_steps(
    ks, emod, ends, hp, s_of, t_tab, w, search, row, chain: bool = True
) -> dict[int, int]:
    """Integer module offset n of every frame of a pass (absolute offset e = emod + n * w).

    Without ``chain``: the row-end anchors interpolated along the pass (a GNSS bias drifts
    slowly), or zero offset without anchors, each frame to the nearest n.

    With ``chain``: one decision for the whole pass (dynamic programming over the frames in
    order along the row). A path of offsets pays for every consecutive pair of frames whose
    module texture it does not line up (correlation below the best step), for every
    frame that disagrees with its own row-end anchor, and a little for change between frames
    and for distance from the offset without ``chain`` (so that, where the texture says
    nothing, the result is the same). A texture link counts only where one step correlates
    clearly better than all others. So consecutive frames stay linked through a quick GNSS
    jump, a wrong anchor is outvoted by the texture links around it, and frames far from the
    row ends get the offset of the anchored frames they are linked to.
    """
    pos = {k: j for j, k in enumerate(ks)}
    anchors = [(pos[k], ends[k]) for k in ks if ends[k] is not None]
    e_pred = {k: 0.0 for k in ks}
    if anchors:
        x, y = np.array(anchors).T
        lo, hi = x < len(ks) / 2, x >= len(ks) / 2
        if lo.any() and hi.any():
            x0, y0 = np.median(x[lo]), np.median(y[lo])
            x1, y1 = np.median(x[hi]), np.median(y[hi])
            e_pred = {k: float(y0 + (pos[k] - x0) * (y1 - y0) / (x1 - x0)) for k in ks}
        else:
            e_pred = {k: float(np.median(y)) for k in ks}
    if not chain or len(ks) < 2:
        return {k: int(np.round((e_pred[k] - emod[k]) / w)) for k in ks}
    tex = {k: _texture(hp[k], s_of[k], t_tab, emod[k], w, row) for k in ks}
    m = int(np.ceil(search / w)) + 1
    states = np.arange(-m, m + 1)
    E = np.array([[emod[k] + q * w for q in states] for k in ks])
    unary = PRIOR_COST * np.abs(E - np.array([e_pred[k] for k in ks])[:, None]) / w
    for j, k in enumerate(ks):
        if ends[k] is not None:
            unary[j] += ANCHOR_COST * (np.abs(E[j] - ends[k]) > w / 2)
    step = states[None, :] - states[:, None]  # from state i (frame j-1) to state l (frame j)
    total, back = unary[0].copy(), []
    for j in range(1, len(ks)):
        c = _texture_match(tex, s_of, emod, ks[j - 1], ks[j], w)
        trans = np.zeros(step.shape)  # no clear texture match: no information
        if np.isfinite(c).sum() >= 3:
            top = np.sort(c[np.isfinite(c)])[::-1]
            best = int(np.nanargmax(c))
            if (
                top[0] > TEXTURE_MIN[0]
                and top[0] - top[1] > TEXTURE_MIN[1]
                and 0 < best < len(c) - 1  # a peak at the edge of the range is a trend
            ):
                half = (len(c) - 1) // 2
                near = np.abs(step) <= half
                trans = np.full(step.shape, np.inf)
                cost = TEXTURE_COST * (top[0] - c)
                trans[near] = np.where(np.isfinite(cost), cost, np.inf)[step[near] + half]
        trans = trans + SMOOTH_COST * np.abs(E[j][None, :] - E[j - 1][:, None]) / w
        cand = total[:, None] + trans
        back.append(np.argmin(cand, axis=0))
        total = cand[back[-1], np.arange(len(states))] + unary[j]
    path = [int(np.argmin(total))]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    return {k: int(states[q]) for k, q in zip(ks, path, strict=True)}


def _refine_row(frames, ks, start, scene, row, search_t, search_s, steps, first: bool):
    """Camera shifts (k -> vector) that align the frames ``ks`` with one row, and the report
    numbers. ``first`` is the pass's first row; later rows start from its correction and only
    look for small residual offsets."""
    a, up = row.along, row.upslope
    w = row.length / row.modules_along
    t = np.arange(-row.width / 2 - search_t - 0.8, row.width / 2 + search_t + 0.8, RES)
    s_all = np.arange(-row.length / 2 - search_s - 12, row.length / 2 + search_s + 12, RES)
    s_of = {k: s_all[np.abs(s_all - (start[k].position - row.center) @ a) < 10] for k in ks}
    table = np.abs(t) < row.width / 2 - 0.1  # along-row profiles: modules only
    poses = dict(start)
    shift = {k: np.zeros(3) for k in ks}
    edge_found, k_med = 1.0, 1.0
    if first and "t" in steps:
        # 1. across-row offset, 2. distance to the module plane (pass median of the scale),
        # 3. across-row offset again (a distance change moves an oblique view across the row)
        et, edge_found = _t_pass(frames, ks, poses, scene, row, s_of, t, search_t)
        if edge_found < 0.5:
            return None, edge_found, 1.0, 0
        if "scale" in steps:
            scale = []
            for k in ks:
                p = poses[k].moved(-et[k] * up)
                o = ortho(frames[k][0], p, scene, row, s_of[k], t)[table]
                scale.append(_s_scale(o, s_of[k], row, (p.position - row.center) @ a))
            k_med = float(np.median(scale))
            for k in ks:
                d = (poses[k].position - row.center) @ row.normal
                shift[k] = shift[k] + (d / k_med - d) * row.normal
                poses[k] = poses[k].moved((d / k_med - d) * row.normal)
            et, _ = _t_pass(frames, ks, poses, scene, row, s_of, t, search_t)
    elif "t" in steps:  # a later row of the pass: small residual across the row
        et = {}
        for k in ks:
            o = ortho(frames[k][0], poses[k], scene, row, s_of[k], t, mask_hidden=False)
            et[k] = _t_fine(o, t, row)
    else:
        et = {k: 0.0 for k in ks}
    for k in ks:
        shift[k] = shift[k] - et[k] * up
    n_anchor = 0
    if "s" in steps:
        # 4. along-row offset modulo one module, and the row ends where visible
        emod, ends, hp = {}, {}, {}
        for k in ks:
            o = ortho(frames[k][0], poses[k].moved(-et[k] * up), scene, row, s_of[k], t)[table]
            emod[k], _ = _s_modulo(o, s_of[k], row)
            hp[k] = _highpass(o)
            ends[k] = None
            if "anchors" in steps:
                ends[k] = _row_end(o, s_of[k], row, search_s)
                if ends[k] is None:
                    ends[k] = _row_end_pattern(o, s_of[k], row, emod[k], search_s)
        n_anchor = sum(e is not None for e in ends.values())
        n = _module_steps(
            ks, emod, ends, hp, s_of, t[table], w, search_s, row, chain="chain" in steps
        )
        for k in ks:
            shift[k] = shift[k] - (emod[k] + n[k] * w) * a
    return shift, edge_found, k_med, n_anchor


def _wide_search(scene: Scene, row, search_t: float) -> float:
    """How far across the row the lower table edge may be looked for when ``search_t`` was not
    enough: up to half the distance to the nearest other row, less a margin, so that the
    search window never reaches the lower edge of a neighbouring row."""
    facing = np.cross(np.array([0.0, 0.0, 1.0]), row.along)  # horizontal, up the slope
    gaps = [abs(float((r.center - row.center) @ facing)) for r in scene.rows if r is not row]
    gaps = [g for g in gaps if g > 0.5]
    if not gaps:
        return 2 * search_t
    return float(min(2 * search_t, max(search_t, 0.5 * min(gaps) - 0.6)))


def refine_poses(
    scene: Scene,
    frames: list[tuple[np.ndarray, Pose]],
    search_t: float = 2.5,
    search_s: float = 4.0,
    steps: tuple[str, ...] = ALL_STEPS,
) -> tuple[list[Pose], list[PassReport]]:
    """Corrected poses for ``frames`` (image, logged pose), in the same order, and a report
    per pass. A pass whose lower table edge is not found within ``search_t`` of where the log
    puts it is searched again up to half the distance to the next row; if the edge is still not
    found, the pass keeps its logged poses.

    The pass's first row fixes the camera shift. When a pass covers several rows, each further
    row gets its own small residual correction (stored in ``Pose.row_shifts``): a heading or
    gimbal error shifts rows at different distances by different amounts, which one camera
    shift cannot absorb. ``steps`` switches parts off for ablation studies.
    """
    lines = scene.extras.get("mission", {}).get("lines", [])
    line_rows = {ln["index"]: [r - 1 for r in ln["rows"]] for ln in lines}
    by_line = defaultdict(list)
    for k, (_, p) in enumerate(frames):
        by_line[p.line].append(k)
    out = [p for _, p in frames]
    reports = []
    for line, ks in sorted(by_line.items()):
        row_ids = line_rows.get(line, [0])
        row = scene.rows[row_ids[0]]
        ks = sorted(ks, key=lambda k: (frames[k][1].position - row.center) @ row.along)
        start = {k: frames[k][1] for k in ks}
        search = search_t
        shift, edge_found, k_med, n_anchor = _refine_row(
            frames, ks, start, scene, row, search, search_s, steps, first=True
        )
        wide = _wide_search(scene, row, search_t)
        if shift is None and wide > search_t:  # look further, but not as far as the next row
            search = wide
            shift, edge_found, k_med, n_anchor = _refine_row(
                frames, ks, start, scene, row, search, search_s, steps, first=True
            )
        if shift is None:
            reports.append(PassReport(line, len(ks), False, edge_found, 1.0, 0, 0.0, search))
            continue
        base = {k: start[k].moved(shift[k]) for k in ks}
        extra = {k: [] for k in ks}
        for ri in row_ids[1:]:
            more, *_ = _refine_row(
                frames, ks, base, scene, scene.rows[ri], search_t, search_s, steps, first=False
            )
            for k in ks:
                extra[k].append((ri, tuple(float(x) for x in more[k])))
        for k in ks:
            out[k] = replace(base[k], row_shifts=tuple(extra[k]))
        dist = [float(np.linalg.norm(shift[k] - (shift[k] @ row.normal) * row.normal)) for k in ks]
        reports.append(
            PassReport(
                line, len(ks), True, edge_found, k_med, n_anchor, float(np.median(dist)), search
            )
        )
    return out, reports
