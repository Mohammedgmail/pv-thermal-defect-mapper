"""Real-flight check of the pose refinement on the PV Hawk test flight (Bommes et al., 2022).

The PV Hawk repository (MIT licence) ships 50 frames of a real radiometric thermal video
(640 x 512 radiometric camera on a drone, consumer GPS, no altitude or attitude log) over two rows
of a plant near Erlangen, with:

* the camera calibration (Brown model) and the GPS track,
* per-frame module outlines from their segmentation network, tracked across frames,
* the modules' georeferenced corners and 19 camera poses from their structure-from-motion
  reconstruction (OpenSfM).

That is everything needed to test the localisation part of this repo on a real flight:

1. A regular plant model (two straight rows, 2P portrait) is fitted to the georeferenced
   modules, as a design layout would give it.
2. The logged poses are what a consumer drone without a gimbal log knows: GPS position,
   a nominal flight height, a nadir camera and the heading of the flight line.
3. The ground truth is independent of all poses: every segmented module outline in every
   frame, with its module ID counted from the row end.

For each frame and pose source (logged, PnP baseline, ours, and the SfM poses as a reference)
the score is the share of segmented modules whose centre the pose puts on the right module
and the corner reprojection error against the segmentation.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/LukasBommes/PV-Hawk
    git -C PV-Hawk sparse-checkout set tests/integration/data      # about 100 MB
    python docs/real_pvhawk.py PV-Hawk/tests/integration/data      # about 10 minutes

Writes docs/results/pvhawk.json and docs/paper/{pvhawk,method_real}.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pvmission.camera import Camera
from pvmission.plant import GeoFrame, Row

from pvdefect.baseline import pnp_refine
from pvdefect.camera import focal_px, project, rays
from pvdefect.refine import ALL_STEPS, _bilinear, refine_poses
from pvdefect.scene import Pose, Scene

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
K_PER_COUNT = 0.04  # raw counts to kelvin, approximate (only the edge thresholds use kelvin)
PIXEL_PITCH_UM = 17.0  # 640 x 512 microbolometer; with the calibrated focal length: 13 mm lens
CLEARANCE_M = 1.5  # assumed height of the row centre above the ground (sets z = 0)


# --------------------------------------------------------------------------- reading the data
def _rot(v) -> np.ndarray:
    """Rotation matrix from an axis-angle vector (OpenSfM convention, world to camera)."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Lens:
    """OpenSfM Brown model; maps between raw pixels and an ideal centred pinhole camera with
    the calibrated focal length (the camera model this repo uses)."""

    def __init__(self, m: dict):
        self.w, self.h = m["width"], m["height"]
        self.size = max(self.w, self.h)
        self.fx, self.fy = m["focal_x"], m["focal_y"]
        self.cx, self.cy = m["c_x"], m["c_y"]
        self.k = (m["k1"], m["k2"], m["k3"])
        self.p = (m["p1"], m["p2"])
        self.f_px = self.fx * self.size

    def _distort(self, x, y):
        k1, k2, k3 = self.k
        p1, p2 = self.p
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        return (
            x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )

    def ideal_to_raw(self, u, v):
        x = (np.asarray(u, float) - (self.w - 1) / 2) / self.f_px
        y = (np.asarray(v, float) - (self.h - 1) / 2) / self.f_px
        xd, yd = self._distort(x, y)
        return (
            (self.fx * xd + self.cx) * self.size + self.w / 2 - 0.5,
            (self.fy * yd + self.cy) * self.size + self.h / 2 - 0.5,
        )

    def raw_to_ideal(self, u, v):
        xd = ((np.asarray(u, float) + 0.5 - self.w / 2) / self.size - self.cx) / self.fx
        yd = ((np.asarray(v, float) + 0.5 - self.h / 2) / self.size - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(20):  # fixed point: undistorted = distorted minus the distortion
            dx, dy = self._distort(x, y)
            x, y = x + (xd - dx), y + (yd - dy)
        return x * self.f_px + (self.w - 1) / 2, y * self.f_px + (self.h - 1) / 2

    def undistort(self, raw: np.ndarray) -> np.ndarray:
        v, u = np.mgrid[0 : self.h, 0 : self.w].astype(float)
        ur, vr = self.ideal_to_raw(u, v)
        out = _bilinear(raw, ur, vr)
        bad = (ur < 0) | (vr < 0) | (ur > self.w - 1) | (vr > self.h - 1)
        return np.where(bad, np.nan, out)


def load(data: Path) -> dict:
    rec = json.loads((data / "mapping/cluster_000000/reconstruction.json").read_text())[0]
    ref = rec["reference_lla"]
    frame = GeoFrame(ref["latitude"], ref["longitude"], 0.0)
    lens = Lens(next(iter(rec["cameras"].values())))
    names = sorted(p.stem for p in (data / "splitted/radiometric").glob("frame_*.tiff"))
    images = [
        np.asarray(Image.open(data / "splitted/radiometric" / f"{n}.tiff"), float) * K_PER_COUNT
        for n in names
    ]
    times = pd.to_datetime((data / "splitted/timestamps.csv").read_text().split())
    gps = json.loads((data / "splitted/gps/gps.json").read_text())
    e, n = frame.to_local([g[1] for g in gps], [g[0] for g in gps])
    sfm = {}
    for key, shot in rec["shots"].items():
        R = _rot(shot["rotation"])
        sfm[Path(key).stem] = (-R.T @ np.asarray(shot["translation"]), R)
    geo = json.loads((data / "mapping/module_geolocations_refined.geojson").read_text())
    corners = {}
    for f in geo["features"]:
        if f["geometry"]["type"] == "Polygon":
            c = np.array(f["geometry"]["coordinates"][0][:4], float)
            ce, cn = frame.to_local(c[:, 1], c[:, 0])
            corners[f["properties"]["track_id"]] = np.stack([ce, cn, c[:, 2]], axis=1)
    quads = defaultdict(dict)
    raw = pickle.loads((data / "quadrilaterals/quadrilaterals.pkl").read_bytes())
    for (track, fname, _), q in raw.items():
        quads[fname][track] = np.asarray(q["quadrilateral"], float)
    return {
        "frame": frame,
        "lens": lens,
        "names": names,
        "images": images,
        "times": times,
        "gps_en": np.stack([e, n], axis=1),
        "sfm": sfm,
        "corners": corners,
        "quads": quads,
    }


# --------------------------------------------------------------------------- plant model
def fit_rows(corners: dict, end_hint: np.ndarray) -> tuple[list[Row], dict, dict]:
    """Two straight 2P rows fitted to the georeferenced module corners, the module ID of
    every mapped track, and the fit numbers.

    Columns are counted from the row end nearest ``end_hint`` (the end the flight sees).
    The SfM map drifts in scale along the rows by a few per cent, so the column of a module
    is its rank along the row, not its distance from the end over the module pitch.
    """
    ids = list(corners)
    P = np.array([corners[i] for i in ids])
    cen = P.mean(axis=1)
    south = cen[:, 1] < np.median(cen[:, 1])
    groups = [south, ~south]  # south row first
    d = []  # the rows are parallel: one direction for both
    for sel in groups:
        c = cen[sel, :2] - np.median(cen[sel, :2], axis=0)
        d.append(np.linalg.svd(c)[2][0])
        d[-1] = d[-1] * np.sign(d[-1] @ d[0])
    heading = float(np.degrees(np.arctan2(*np.mean(d, axis=0))) % 360)
    rows, track_id, info = [], {}, {}
    for index, sel in enumerate(groups):
        Q = P[sel].reshape(-1, 3)
        keep = np.ones(len(Q), bool)  # plane fit robust to badly triangulated corners
        for _ in range(3):
            c0 = Q[keep].mean(axis=0)
            n = np.linalg.svd(Q[keep] - c0)[2][2]
            n = n * np.sign(n[2])
            r = (Q - c0) @ n
            keep = np.abs(r) < max(3 * np.median(np.abs(r[keep])), 0.05)
        tilt = float(np.degrees(np.arccos(n[2])))
        facing = float(np.degrees(np.arctan2(n[0], n[1])) % 360)
        azimuth = min(
            ((heading + k) % 360 for k in (90.0, -90.0)),
            key=lambda az: abs((az - facing + 180) % 360 - 180),
        )
        ref = Row(index, c0, 1.0, 1.0, tilt, azimuth, 1, 2)
        a, up = ref.along, ref.upslope
        Ps, cs = P[sel], cen[sel]
        s, t = (cs - c0) @ a, (cs - c0) @ up
        height = float(np.median(np.abs((Ps[:, 1] - Ps[:, 0]) @ up)))  # module long side
        level = (t > np.median(t)).astype(int) + 1
        s_hint = float((np.r_[end_hint, c0[2]] - c0) @ a)
        sign = -1.0 if abs(s_hint - s.min()) < abs(s_hint - s.max()) else 1.0  # +1: end at max s
        sid = np.array(ids)[sel]
        pitch, ranks = [], {}
        for lvl in (1, 2):
            m = level == lvl
            o = np.argsort(-sign * s[m])  # nearest the seen end first
            pitch += list(np.abs(np.diff(s[m][o])))
            for rank, j in enumerate(o, start=1):
                ranks[sid[m][j]] = (rank, lvl)
        pitch = float(np.median([x for x in pitch if 0.8 < x < 1.4]))
        n_along = max(rk for rk, _ in ranks.values()) + 8  # the row goes on beyond the map
        length = n_along * pitch
        end = (s.max() if sign > 0 else s.min()) + sign * pitch / 2
        center = c0 + (end - sign * length / 2) * a + float(np.median(t)) * up
        rows.append(Row(index, center, length, 2 * height, tilt, azimuth, n_along, 2))
        for tid, (rank, lvl) in ranks.items():
            col = n_along - rank + 1 if sign > 0 else rank
            track_id[tid] = f"R{index + 1:02d}-C{col:03d}-{lvl}"
        info[f"row{index + 1}"] = {
            "modules_mapped": int(sel.sum()),
            "tilt_deg": round(tilt, 1),
            "azimuth_deg": round(azimuth, 1),
            "module_pitch_m": round(pitch, 3),
            "module_long_side_m": round(height, 3),
            "plane_rms_m": round(float(np.sqrt(np.mean(r[keep] ** 2))), 3),
            "corners_off_plane_0p3m": int((np.abs(r) > 0.3).sum()),
            "seen_end": "max s" if sign > 0 else "min s",
        }
    z0 = float(np.mean([r.center[2] for r in rows])) - CLEARANCE_M  # ground at z = 0
    rows = [
        Row(
            r.index,
            r.center - [0, 0, z0],
            r.length,
            r.width,
            r.tilt,
            r.azimuth,
            r.modules_along,
            r.modules_across,
        )
        for r in rows
    ]
    info["z_offset_m"] = round(z0, 3)
    return rows, track_id, info


# --------------------------------------------------------------------------- scoring
class SfMCamera:
    """An SfM pose (camera centre, world-to-camera rotation) with the ideal pinhole camera."""

    def __init__(self, C: np.ndarray, R: np.ndarray, cam: Camera, z0: float):
        self.C, self.R, self.cam = C - [0, 0, z0], R, cam

    def for_row(self, row_index: int) -> SfMCamera:
        return self

    def project(self, P):
        Pc = (np.asarray(P, float) - self.C) @ self.R.T
        f = focal_px(self.cam)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (self.cam.width_px - 1) / 2 + f * Pc[..., 0] / Pc[..., 2]
            v = (self.cam.height_px - 1) / 2 + f * Pc[..., 1] / Pc[..., 2]
        return u, v, Pc[..., 2]

    def ray(self, u, v):
        f = focal_px(self.cam)
        x = (u - (self.cam.width_px - 1) / 2) / f
        y = (v - (self.cam.height_px - 1) / 2) / f
        d = self.R.T @ np.array([x, y, 1.0])
        return self.C, d / np.linalg.norm(d)


class PoseCamera:
    """A Pose of this repo with the scene camera, in the same interface as SfMCamera."""

    def __init__(self, pose: Pose, cam: Camera):
        self.pose, self.cam = pose, cam

    def for_row(self, row_index: int) -> PoseCamera:
        return PoseCamera(self.pose.for_row(row_index), self.cam)

    def project(self, P):
        return project(P, self.pose, self.cam)

    def ray(self, u, v):
        return self.pose.position, rays(np.array(u), np.array(v), self.pose, self.cam)


def module_at(scene: Scene, camera, u: float, v: float) -> str | None:
    """The module of the plant model that ``camera`` puts at ideal pixel (u, v)."""
    hits = []
    for row in scene.rows:
        C, d = camera.for_row(row.index).ray(u, v)
        denom = d @ row.normal
        if abs(denom) < 1e-9:
            continue
        lam = ((row.center - C) @ row.normal) / denom
        X = C + lam * d
        mid = row.module_id(
            float((X - row.center) @ row.along), float((X - row.center) @ row.upslope)
        )
        if mid is not None and lam > 0:
            hits.append((lam, mid))
    return min(hits)[1] if hits else None


def score_frame(scene: Scene, camera, quads: dict, track_id: dict, lens: Lens) -> dict:
    """Share of the segmented modules whose centre the camera puts on the right module, and
    the median corner error (raw pixels) of the outlines against the projected modules."""
    right, total, err = 0, 0, []
    for tid, q in quads.items():
        if tid not in track_id:
            continue
        uq, vq = lens.raw_to_ideal(q[:, 0], q[:, 1])
        total += 1
        right += module_at(scene, camera, float(np.mean(uq)), float(np.mean(vq))) == track_id[tid]
        m = scene.module(track_id[tid])
        u, v, _ = camera.for_row(m.row.index).project(m.corners())
        ur, vr = lens.ideal_to_raw(u, v)
        D = np.hypot(ur[:, None] - q[None, :, 0], vr[:, None] - q[None, :, 1])
        err.append(float(np.mean(D.min(axis=1))))  # corner order differs: nearest corner
    return {"modules": total, "right": right, "corner_px": float(np.median(err)) if err else None}


# --------------------------------------------------------------------------- main
METHODS = {  # label: refinement steps (None: no refinement)
    "logged": None,
    "PnP baseline": "pnp",
    "ours, no row ends": ("t", "scale", "s"),
    "ours, no texture links": ("t", "scale", "s", "anchors"),
    "ours": ALL_STEPS,
}


def setup(data: Path, height: float):
    d = load(data)
    rows, track_id, info = fit_rows(d["corners"], d["gps_en"][-1])
    lens = d["lens"]
    cam = Camera(
        "640 x 512 thermal, 13 mm (calibrated)",
        lens.w,
        lens.h,
        PIXEL_PITCH_UM,
        lens.f_px * PIXEL_PITCH_UM * 1e-3,
    )
    # logged poses: GPS, a nominal flight height, nadir, the heading of the flight line
    en = d["gps_en"]
    slope = np.polyfit(en[:, 0], en[:, 1], 1)[0]
    yaw = float(np.degrees(np.arctan2(1.0, slope))) + 180.0  # image x points north
    logged = [
        Pose(np.array([e, n, height]), yaw, -90.0, t, 1, k)
        for k, ((e, n), t) in enumerate(zip(en, d["times"], strict=True))
    ]
    mission = {"lines": [{"index": 1, "rows": [1, 2], "speed_mps": 4.0}]}
    scene = Scene("PV Hawk test flight", d["frame"], "Europe/Berlin", rows, cam, logged)
    scene.extras["mission"] = mission
    images = [lens.undistort(im) for im in d["images"]]
    return d, scene, images, track_id, info


def run(d, scene, images, track_id, info, methods=METHODS) -> tuple[pd.DataFrame, dict]:
    cam, lens, z0 = scene.camera, d["lens"], info["z_offset_m"]
    frames = list(zip(images, scene.poses, strict=True))
    poses, reports = {}, {}
    for label, steps in methods.items():
        if steps is None:
            poses[label] = list(scene.poses)
        elif steps == "pnp":
            poses[label] = pnp_refine(scene, frames)
        else:
            poses[label], rep = refine_poses(scene, frames, steps=steps)
            reports[label] = [r.as_dict() for r in rep]
    out = []
    for k, name in enumerate(d["names"]):
        q = d["quads"].get(name, {})
        for label, ps in poses.items():
            out.append(
                {
                    "frame": k,
                    "method": label,
                    **score_frame(scene, PoseCamera(ps[k], cam), q, track_id, lens),
                }
            )
        if name in d["sfm"]:
            C, R = d["sfm"][name]
            sfm = SfMCamera(C, R, cam, z0)
            out.append(
                {"frame": k, "method": "SfM poses", **score_frame(scene, sfm, q, track_id, lens)}
            )
    return pd.DataFrame(out), {"reports": reports, "poses": poses}


def summary(df: pd.DataFrame) -> dict:
    g = df.groupby("method", sort=False).agg(
        modules=("modules", "sum"), right=("right", "sum"), corner_px=("corner_px", "median")
    )
    return {
        m: {
            "modules": int(r.modules),
            "right": int(r.right),
            "share_right": round(r.right / r.modules, 4),
            "corner_px_median": round(float(r.corner_px), 2),
            "frames_all_right": int((df[df.method == m].right == df[df.method == m].modules).sum()),
        }
        for m, r in g.iterrows()
    }


def contrast_stats(images, poses, scene, row) -> dict:
    """Unit-free image statistics that the refinement depends on, median over frames: the
    step from the ground in front of the table up to the modules (S), the depth of the frame
    lines between modules (D) and the spread of temperatures inside modules (sigma)."""
    from pvdefect.refine import RES, _dips, _nanmedian, ortho

    w, h = row.length / row.modules_along, row.width / row.modules_across
    t = np.arange(-row.width / 2 - 1.2, row.width / 2 + 0.3, RES)
    out = {"S": [], "D": [], "sigma": []}
    for img, p in zip(images, poses, strict=True):
        p = p.for_row(row.index)
        sc = (p.position - row.center) @ row.along
        s = np.arange(sc - 8, sc + 8, RES)
        s = s[np.abs(s) < row.length / 2 - w]  # modules only, away from the row ends
        if len(s) < int(4 * w / RES):
            continue
        o = ortho(img, p, scene, row, s, t)
        table = np.abs(t) < row.width / 2 - 0.2
        ground = (t > -row.width / 2 - 1.0) & (t < -row.width / 2 - 0.3)
        out["S"].append(float(np.nanmedian(o[table]) - np.nanmedian(o[ground])))
        prof = _nanmedian(o[table], axis=0)
        lines = -row.length / 2 + np.arange(row.modules_along + 1) * w
        lines = lines[(lines > s[0] + 0.5) & (lines < s[-1] - 0.5)]
        out["D"].append(float(np.nanmedian(_dips(prof, s, lines))))
        xs = (s + row.length / 2) % w
        yt = (t + row.width / 2) % h
        inner = (yt[:, None] > 0.1) & (yt[:, None] < h - 0.1) & (xs[None, :] > 0.1)
        inner &= (xs[None, :] < w - 0.1) & table[:, None]
        col = (
            np.floor((s + row.length / 2) / w)[None, :]
            + 1000 * np.floor((t + row.width / 2) / h)[:, None]
        )
        res = []
        for c in np.unique(col[inner]):
            v = o[inner & (col == c)]
            v = v[np.isfinite(v)]
            if v.size > 50:
                res.append(v - np.median(v))
        if res:
            r = np.concatenate(res)
            out["sigma"].append(float(1.4826 * np.median(np.abs(r))))
    med = {k: float(np.median(v)) for k, v in out.items() if v}
    med["D_over_S"] = med["D"] / med["S"]
    med["sigma_over_S"] = med["sigma"] / med["S"]
    return {k: round(v, 3) for k, v in med.items()}


def twin_stats() -> dict:
    """The same statistics for one pass of the thermal twin (noon plan, true poses)."""
    from pvdefect.simulate import ThermalTwin, flown_poses, random_defects

    scene = Scene.from_mission(ROOT / "examples" / "june-noon" / "mission.json")
    defects = random_defects(scene, np.random.default_rng(7))
    true = flown_poses(scene, np.random.default_rng(7))
    twin = ThermalTwin(scene, defects, seed=7)
    ps = [p for p in true if p.line == 3][4:-4]
    imgs = [twin.render(p) for p in ps]
    return contrast_stats(imgs, ps, scene, scene.rows[3])


def figure_method(d, scene, images, extra, path: Path, frame: int = 44) -> None:
    """How the refinement reads a real frame: (a) the frame resampled onto the row plane with
    the refined pose and the plant model's module lattice; (b) the profile across the table;
    (c) frame-line dips along the row near the row end; (d) texture match of two consecutive
    frames for each module step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pvdefect.refine import RES, _dips, _nanmedian, _texture, _texture_match, ortho

    row = scene.rows[0]
    w = row.length / row.modules_along
    p = extra["poses"]["ours"][frame].for_row(0)
    sc = (p.position - row.center) @ row.along
    s = np.arange(sc - 7, sc + 7, RES)
    t = np.arange(-row.width / 2 - 1.5, row.width / 2 + 1.5, RES)
    o = ortho(images[frame], p, scene, row, s, t, mask_hidden=False)
    fig, axes = plt.subplots(
        1, 4, figsize=(7.2, 1.9), gridspec_kw={"width_ratios": [1.6, 0.8, 1.2, 0.9]}
    )
    ax = axes[0]
    lo, hi = np.nanpercentile(o, [2, 99.5])
    ax.imshow(o, origin="lower", extent=[s[0], s[-1], t[0], t[-1]], cmap="inferno",
              vmin=lo, vmax=hi, aspect="auto")  # fmt: skip
    for x in -row.length / 2 + np.arange(row.modules_along + 1) * w:
        if s[0] < x < s[-1]:
            ax.plot([x, x], [-row.width / 2, row.width / 2], color="#2a78d6", lw=0.5)
    for y in (-row.width / 2, 0, row.width / 2):
        ax.plot([max(s[0], -row.length / 2), s[-1]], [y, y], color="#2a78d6", lw=0.5)
    ax.set_xlabel("along the row, s (m)", fontsize=6.5)
    ax.set_ylabel("up the slope, t (m)", fontsize=6.5)
    ax.set_title("(a) Real frame on the row plane", loc="left", fontsize=7)
    ax = axes[1]
    prof = _nanmedian(o, axis=1)
    ax.plot(prof, t, color="#0b0b0b", lw=1)
    ax.axhline(-row.width / 2, color="#2a78d6", lw=0.6)
    ax.axhline(row.width / 2, color="#2a78d6", lw=0.6)
    ax.set_xlabel("K (approx.)", fontsize=6.5)
    ax.set_title("(b) Across", loc="left", fontsize=7)
    ax = axes[2]
    table = np.abs(t) < row.width / 2 - 0.1
    alongp = _nanmedian(o[table], axis=0)
    ax.plot(s, alongp, color="#0b0b0b", lw=0.8)
    lat = -row.length / 2 + np.arange(-4, 12) * w
    lat = lat[(lat > s[0] + 0.4) & (lat < s[-1] - 0.4)]
    dv = _dips(alongp, s, lat)
    ax2 = ax.twinx()
    ax2.bar(lat, dv, width=0.25, color=np.where(lat > -row.length / 2 + 0.1, "#2a78d6", "#eb6834"))
    ax2.set_ylim(np.nanmin(dv) * 3, 0.5)
    ax2.tick_params(labelsize=5.5)
    ax.axvline(-row.length / 2, color="#1baf7a", lw=0.8, ls="--")
    ax.set_xlabel("s (m)", fontsize=6.5)
    ax.set_title("(c) Row end: frame lines stop", loc="left", fontsize=7)
    ax = axes[3]
    ks = [k for k in range(len(images)) if k != frame][:0] or [frame - 1, frame]
    hp, s_of, emod = {}, {}, {}
    from pvdefect.refine import _highpass

    for k in ks:
        pk = extra["poses"]["ours"][k].for_row(0)
        sk = np.arange((pk.position - row.center) @ row.along - 7,
                       (pk.position - row.center) @ row.along + 7, RES)  # fmt: skip
        ok = ortho(images[k], pk, scene, row, sk, t)[table]
        hp[k], s_of[k], emod[k] = _highpass(ok), sk, 0.0
    tex = {k: _texture(hp[k], s_of[k], t[table], 0.0, w, row) for k in ks}
    c = _texture_match(tex, s_of, emod, ks[0], ks[1], w)
    ax.bar(np.arange(-2, 3), c, color=["#b5b3ad"] * 2 + ["#2a78d6"] + ["#b5b3ad"] * 2)
    ax.set_xlabel("module step", fontsize=6.5)
    ax.set_ylabel("texture correlation", fontsize=6.5)
    ax.set_title("(d) Frame link", loc="left", fontsize=7)
    for a in axes:
        a.tick_params(labelsize=5.5)
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figure(d, scene, images, track_id, df, extra, path: Path, frame: int | None = None) -> None:
    """(a) A frame where the logged poses put the modules on the wrong IDs, with the plant
    model projected from the logged and the refined pose; (b) share right per frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens, cam = d["lens"], scene.camera
    if frame is None:  # most modules moved from wrong to right by the refinement
        share = df.pivot_table(index="frame", columns="method", values="right", aggfunc="sum")
        frame = int((share["ours"] - share["logged"]).idxmax())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    raw = d["images"][frame]
    ax.imshow(raw, cmap="gray", vmin=np.percentile(raw, 1), vmax=np.percentile(raw, 99.5))
    for q in d["quads"][d["names"][frame]].values():
        ax.fill(q[:, 0], q[:, 1], color="#1baf7a", alpha=0.25, lw=0)
    for label, color in (("logged", "#eb6834"), ("ours", "#2a78d6")):
        pc = PoseCamera(extra["poses"][label][frame], cam)
        for m in scene.modules():
            u, v, dep = pc.for_row(m.row.index).project(m.corners())
            if (dep <= 0).any():
                continue
            ur, vr = lens.ideal_to_raw(u, v)
            if ur.min() < -20 or vr.min() < -20 or ur.max() > lens.w + 20 or vr.max() > lens.h + 20:
                continue
            ax.plot(np.r_[ur, ur[:1]], np.r_[vr, vr[:1]], color=color, lw=0.7, alpha=0.9)
    ax.set_xlim(0, lens.w - 1)
    ax.set_ylim(lens.h - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) Frame {frame}: plant model on the real image", loc="left", fontsize=8)
    ax.fill([], [], color="#1baf7a", alpha=0.4, label="segmented modules")
    ax.plot([], [], color="#eb6834", label="model, logged GPS pose")
    ax.plot([], [], color="#2a78d6", label="model, refined pose")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=5.5, frameon=False)
    ax = axes[1]
    for label, color, mk in (
        ("logged", "#b5b3ad", "-"),
        ("PnP baseline", "#eb6834", "-"),
        ("ours", "#2a78d6", "-"),
        ("SfM poses", "#008300", "o"),
    ):
        sub = df[df.method == label]
        share = sub.right / sub.modules.clip(lower=1)
        if mk == "o":
            ax.plot(sub.frame, share, "o", color=color, ms=2.5, label=label)
        else:
            ax.plot(sub.frame, share, mk, color=color, lw=1.2, label=label)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("frame (7.7 per second)", fontsize=7)
    ax.set_ylabel("modules given the right ID", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("(b) Share right per frame", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path, help="PV-Hawk/tests/integration/data")
    ap.add_argument("--height", type=float, default=20.0, help="nominal flight height (m)")
    a = ap.parse_args()
    d, scene, images, track_id, info = setup(a.data, a.height)
    df, extra = run(d, scene, images, track_id, info)
    result = {
        "data": "PV Hawk integration-test flight (Bommes et al., 2022; MIT licence)",
        "frames": len(images),
        "nominal_height_m": a.height,
        "plant_model": info,
        "summary": summary(df),
        "refine_reports": extra["reports"],
    }
    sens = {}
    for h in (17.0, 18.0, 22.0, 23.0):  # how much the nominal height may be off
        d2, sc2, im2, tid2, inf2 = setup(a.data, h)
        df2, _ = run(d2, sc2, im2, tid2, inf2, {"logged": None, "ours": ALL_STEPS})
        sens[f"{h:g}"] = summary(df2)
    result["height_sensitivity"] = sens
    real = contrast_stats(images, extra["poses"]["ours"], scene, scene.rows[0])
    result["contrast"] = {
        f"real (kelvin at {K_PER_COUNT:.2f} K per count)": real,
        "twin (noon plan, row 4)": twin_stats(),
    }
    result["per_frame"] = df.to_dict(orient="records")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pvhawk.json").write_text(json.dumps(result, indent=1))
    figure(d, scene, images, track_id, df, extra, ROOT / "docs" / "paper" / "pvhawk")
    figure_method(d, scene, images, extra, ROOT / "docs" / "paper" / "method_real")
    print(
        json.dumps({k: result[k] for k in ("summary", "height_sensitivity", "contrast")}, indent=1)
    )


if __name__ == "__main__":
    main()
