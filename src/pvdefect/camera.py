"""Pinhole projection between the world (ENU) and image pixels for a camera pose.

Image convention (matches pvmission): x to the right, y down, the long image side along the
row. Pixel (0, 0) is the centre of the top-left pixel.
"""

from __future__ import annotations

import numpy as np
from pvmission.camera import Camera

from .scene import Pose


def focal_px(cam: Camera) -> float:
    return cam.focal_mm / (cam.pixel_pitch_um * 1e-3)


def project(
    points: np.ndarray, pose: Pose, cam: Camera
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and depth along the optical axis for world points (..., 3)."""
    x, y, z = pose.axes()
    d = np.asarray(points, float) - pose.position
    depth = d @ z
    f = focal_px(cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (cam.width_px - 1) / 2 + f * (d @ x) / depth
        v = (cam.height_px - 1) / 2 + f * (d @ y) / depth
    return u, v, depth


def rays(u, v, pose: Pose, cam: Camera) -> np.ndarray:
    """Unit world directions (..., 3) through pixel coordinates (u, v)."""
    x, y, z = pose.axes()
    f = focal_px(cam)
    a = (np.asarray(u, float) - (cam.width_px - 1) / 2) / f
    b = (np.asarray(v, float) - (cam.height_px - 1) / 2) / f
    d = a[..., None] * x + b[..., None] * y + z
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def pixel_grid(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(cam.width_px, dtype=float), np.arange(cam.height_px, dtype=float))
