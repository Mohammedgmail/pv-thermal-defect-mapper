"""First-order model: how camera pose errors move the image on the module plane, and what that
does to module- and cell-level mapping.

For a pixel, the logged pose sees a different point of the module plane than the true pose.
To first order, the offset in row coordinates (s along the row, t up the slope) is

    [ds, dt] = J @ [dx, dy, dz, dyaw, dpitch]

where J is the 2 x 5 Jacobian of the plane point with respect to the pose (east, north, up
in metres; heading and gimbal pitch in radians). Its size follows the geometry:

- a heading error moves the image along the row by about the horizontal range;
- a pitch error moves it up the slope by about the slant range divided by the sine of the
  angle between the line of sight and the module plane;
- a position error parallel to the plane moves it by the same amount;
- a height error moves it along the line of sight.

A hot spot at a uniformly random place inside a cell of width ``w`` stays in that cell under an
offset d with probability max(0, 1 - |d| / w), per axis. Averaged over a zero-mean Gaussian
offset with standard deviation sigma, that is

    p(sigma, w) = 2 Phi(w / sigma) - 1 - sqrt(2 / pi) (sigma / w) (1 - exp(-w^2 / (2 sigma^2)))

and the chance of the right cell is p_s * p_t, treating the two axes as independent. The same
formula with the module size gives the chance of the right module.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .camera import rays
from .scene import Pose, Scene

PARAMS = ("east_m", "north_m", "up_m", "yaw_rad", "pitch_rad")


def plane_point(pose: Pose, scene: Scene, row, u: float, v: float) -> np.ndarray:
    """Where the ray through pixel (u, v) meets the plane of ``row``."""
    d = rays(np.array([u]), np.array([v]), pose, scene.camera)[0]
    t = ((row.center - pose.position) @ row.normal) / (d @ row.normal)
    return pose.position + t * d


def jacobian(pose: Pose, scene: Scene, row, u: float | None = None, v: float | None = None):
    """2 x 5 Jacobian of (s, t) on the row plane with respect to the pose (see module doc),
    by central differences at pixel (u, v) (default: the image centre)."""
    cam = scene.camera
    u = (cam.width_px - 1) / 2 if u is None else u
    v = (cam.height_px - 1) / 2 if v is None else v
    J = np.zeros((2, 5))
    steps = (0.01, 0.01, 0.01, np.radians(0.01), np.radians(0.01))
    for j, h in enumerate(steps):
        dp, dyaw, dpitch = np.zeros(3), 0.0, 0.0
        if j < 3:
            dp[j] = h
        elif j == 3:
            dyaw = np.degrees(h)
        else:
            dpitch = np.degrees(h)
        hi = plane_point(pose.moved(dp, dyaw, dpitch), scene, row, u, v)
        lo = plane_point(pose.moved(-dp, -dyaw, -dpitch), scene, row, u, v)
        diff = (hi - lo) / (2 * h)
        J[:, j] = diff @ row.along, diff @ row.upslope
    return J


def offset_sd(J: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standard deviations of (ds, dt) for independent pose errors with standard deviations
    ``sd`` (same order as PARAMS)."""
    cov = J @ np.diag(np.asarray(sd, float) ** 2) @ J.T
    return np.sqrt(np.diag(cov))


def p_same_cell(sigma, width) -> np.ndarray:
    """Chance that a point uniformly placed in an interval of ``width`` stays inside it under a
    Gaussian offset with standard deviation ``sigma`` (see module doc)."""
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    r = width / sigma
    return 2 * norm.cdf(r) - 1 - np.sqrt(2 / np.pi) / r * (1 - np.exp(-(r**2) / 2))


def p_right_cell(sd_st, cell) -> float:
    """Chance that a hot spot lands on the right cell (or module): ``sd_st`` are the offset
    standard deviations along the row and up the slope, ``cell`` the cell (or module) size."""
    return float(p_same_cell(sd_st[0], cell[0]) * p_same_cell(sd_st[1], cell[1]))


def pose_sd(errors) -> np.ndarray:
    """Total pose error standard deviations (bias and jitter added in quadrature) of a
    :class:`pvdefect.simulate.PoseErrors`, in PARAMS order."""
    e = errors
    h = np.hypot(e.position_bias_m, e.position_jitter_m)
    return np.array(
        [
            h,
            h,
            1.5 * h,
            np.radians(np.hypot(e.yaw_bias_deg, e.yaw_jitter_deg)),
            np.radians(np.hypot(e.pitch_bias_deg, e.pitch_jitter_deg)),
        ]
    )
