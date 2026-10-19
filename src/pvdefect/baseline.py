"""Baseline pose correction: fit the camera pose to the module grid seen in the image.

This is the standard way to register an image to a known structure (model-based tracking; the
perspective-n-point step of Kozák et al., 2025, without their row-end initialisation).

1. Project the module corners (crossings of the cool frame lines) with the logged pose.
2. Look for the actual crossing near each one.
3. Fit the camera position, heading and gimbal pitch to these matches with a robust least
   squares fit.
4. Repeat with a smaller search window.

The module grid repeats every module. A pose error of more than half a module spacing makes
each corner match its neighbour, and the fit then converges to a pose one module off. The
row-end anchors in :mod:`pvdefect.refine` exist to remove exactly this failure.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .camera import project
from .scene import Pose, Scene


def _coolness(image: np.ndarray) -> np.ndarray:
    """Positive on thin lines cooler than their surroundings (the aluminium frames), crossings
    included: a black top-hat (grey closing minus image)."""
    return ndimage.grey_closing(image, size=(5, 5)) - image


def _grid(row) -> np.ndarray:
    """World points of all module-corner crossings of a row, shape (n, 3)."""
    s = -row.length / 2 + np.arange(row.modules_along + 1) * row.length / row.modules_along
    t = -row.width / 2 + np.arange(row.modules_across + 1) * row.width / row.modules_across
    S, T = np.meshgrid(s, t)
    return row.point(S.ravel(), T.ravel())


def _directions(P, row, pose, cam):
    """Unit image directions of the row axis and the slope axis at each point."""
    u0, v0, _ = project(P, pose, cam)
    out = []
    for axis in (row.along, row.upslope):
        u1, v1, _ = project(P + 0.2 * axis, pose, cam)
        d = np.stack([u1 - u0, v1 - v0], axis=-1)
        out.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
    return out


def _detect(cool, uv, dirs, radius: float, arm: float = 6.0, threshold: float = 1.0):
    """Best crossing within ``radius`` px of each predicted point, or NaN if none is clear.

    The response of a candidate is the mean coolness along two short segments through it,
    along the projected row and slope directions.
    """
    offs = np.arange(-radius, radius + 1e-9, 1.0)
    du, dv = np.meshgrid(offs, offs)
    keep = du**2 + dv**2 <= radius**2
    du, dv = du[keep], dv[keep]
    steps = np.linspace(-arm, arm, 2 * int(arm) + 1)
    found = np.full((len(uv), 2), np.nan)
    for i, ((u, v), da, db) in enumerate(zip(uv, dirs[0], dirs[1], strict=True)):
        cu, cv = u + du, v + dv  # candidates
        arms = []
        for d in (da, db):
            su = cu[:, None] + steps[None, :] * d[0]
            sv = cv[:, None] + steps[None, :] * d[1]
            arms.append(
                ndimage.map_coordinates(cool, [sv, su], order=1, mode="constant", cval=0.0).mean(1)
            )
        resp = np.minimum(*arms)  # a crossing needs both lines
        j = int(np.argmax(resp))
        if resp[j] > threshold:
            found[i] = (cu[j], cv[j])
    return found


def _apply(pose: Pose, x) -> Pose:
    return pose.moved(x[:3], x[3], x[4])


def pnp_refine(
    scene: Scene, frames: list[tuple[np.ndarray, Pose]], radius_fraction: float = 0.45
) -> list[Pose]:
    """Poses fitted to the module grid, frame by frame, starting from the logged poses.

    The first search window is ``radius_fraction`` of the module spacing along the row, in
    pixels. Wider windows reach further but match neighbouring corners more often.
    """
    cam = scene.camera
    lines = scene.extras.get("mission", {}).get("lines", [])
    line_rows = {ln["index"]: [r - 1 for r in ln["rows"]] for ln in lines}
    out = []
    for image, logged in frames:
        rows = [scene.rows[i] for i in line_rows.get(logged.line, range(len(scene.rows)))]
        P = np.concatenate([_grid(r) for r in rows])
        owner = np.concatenate([[k] * len(_grid(r)) for k, r in enumerate(rows)])
        cool = _coolness(np.asarray(image, float))
        x = np.zeros(5)
        # module spacing along the row in pixels at the image centre
        u, v, depth = project(P, logged, cam)
        inside = (depth > 0) & (u > 8) & (v > 8) & (u < cam.width_px - 9) & (v < cam.height_px - 9)
        if inside.sum() < 6:
            out.append(logged)
            continue
        r0 = rows[0]
        w = r0.length / r0.modules_along
        uc, vc, _ = project(np.array([r0.center, r0.center + w * r0.along]), logged, cam)
        spacing = float(np.hypot(uc[1] - uc[0], vc[1] - vc[0]))
        pose = logged
        for radius in (radius_fraction * spacing, 0.2 * spacing, 0.1 * spacing):
            u, v, depth = project(P, pose, cam)
            ok = (depth > 0) & (u > 8) & (v > 8) & (u < cam.width_px - 9) & (v < cam.height_px - 9)
            if ok.sum() < 6:
                break
            dirs = [np.zeros((ok.sum(), 2)), np.zeros((ok.sum(), 2))]
            for k, r in enumerate(rows):
                sel = owner[ok] == k
                if sel.any():
                    da, db = _directions(P[ok][sel], r, pose, cam)
                    dirs[0][sel], dirs[1][sel] = da, db
            found = _detect(cool, np.c_[u[ok], v[ok]], dirs, radius)
            good = np.isfinite(found[:, 0])
            if good.sum() < 6:
                break
            Pg, xy = P[ok][good], found[good]

            def resid(p, Pg=Pg, xy=xy, start=logged):
                uu, vv, _ = project(Pg, _apply(start, p), cam)
                return np.r_[uu - xy[:, 0], vv - xy[:, 1]]

            fit = least_squares(
                resid, x, loss="huber", f_scale=1.0, x_scale=[0.5, 0.5, 0.5, 0.5, 0.5]
            )
            x = fit.x
            pose = _apply(logged, x)
        out.append(pose)
    return out
