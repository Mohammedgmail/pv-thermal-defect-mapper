"""Thermal digital twin: render the frames a drone would take, with known defects.

Real inspection data with exact module-level ground truth and camera poses is not public,
so the mapper is evaluated on frames rendered from the flight plan itself. The renderer
ray-casts every pixel into the plant and models:

* module temperature from clear-sky plane-of-array irradiance, air temperature and wind
  (pvlib Faiman model), a 1 K bottom-to-top gradient, cell-to-cell scatter and cooler
  aluminium frames;
* defects in module coordinates: hot cell(s), an overheated bypass-diode substring (one third
  of the module, full length) and offline modules (whole module warmer);
* sunlit and row-shaded ground with a fixed texture;
* the sun's mirror reflection in the glass, a bright patch whose position depends on the
  viewing direction (this is what the planner's glare margin keeps out of the images);
* lower apparent temperature at oblique viewing angles, lens blur, motion blur, sensor noise
  (NETD) and vignetting;
* camera pose errors: the image is rendered from the true pose, the mapper only gets a
  logged pose with GNSS and gimbal errors.

The thermal and optical parameters are plausible round numbers, not measurements; see the
README.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from PIL import Image
from pvmission.plant import Plant
from pvmission.sun import clear_sky_poa, sun_vector
from scipy import ndimage

from .camera import pixel_grid, rays
from .scene import FRAME_M, Module, Pose, Scene

KINDS = ("hot_cell", "hot_cells", "substring", "offline", "string")
NUISANCES = ("soiling",)  # warm patterns that are not module faults (rendered, never scored)


@dataclass
class Defect:
    module_id: str
    kind: str  # one of KINDS
    delta_t: float  # K above the healthy module
    cells: list[tuple[int, int]] = field(default_factory=list)  # (row 0-11, column 0-5)
    substring: int = -1  # 0..2


@dataclass(frozen=True)
class Weather:
    air_c: float = 21.0
    wind_ms: float = 3.0


@dataclass(frozen=True)
class Optics:
    netd_k: float = 0.05  # noise-equivalent temperature difference (H20T spec: <= 50 mK)
    blur_px: float = 0.6  # Gaussian lens blur (sigma)
    vignetting_k: float = 0.3  # corner darkening
    reflection_k: float = 20.0  # peak apparent rise where the sun is mirrored into the lens
    reflection_deg: float = 2.5  # angular width of that reflection
    frame_k: float = -6.0  # aluminium frames look cooler (low emissivity)


@dataclass(frozen=True)
class ModuleThermal:
    """Temperature pattern of a healthy module (the defaults are a clean module).

    Real modules also show cell-to-cell mismatch of up to about a kelvin, a warm patch over
    the junction box and cooler edges next to the frame; the stress tests turn these up.
    """

    cell_sigma_k: float = 0.25  # cell-to-cell scatter (standard deviation)
    jbox_k: float = 0.0  # warm patch over the junction box (behind the top of the module)
    edge_k: float = 0.0  # change within a few cm of the frame (negative: cooler)


@dataclass(frozen=True)
class PoseErrors:
    """Logged pose minus true pose: a per-flight bias plus independent per-image jitter.

    GNSS error is mostly a slowly varying bias (about a metre or two without RTK, a couple
    of centimetres with RTK) with little jitter from one image to the next.
    """

    position_bias_m: float = 0.02  # horizontal, per flight (RTK by default)
    position_jitter_m: float = 0.02  # horizontal, per image
    yaw_bias_deg: float = 0.3
    yaw_jitter_deg: float = 0.3
    pitch_bias_deg: float = 0.2
    pitch_jitter_deg: float = 0.2


RTK = PoseErrors()
GNSS = PoseErrors(position_bias_m=1.5, position_jitter_m=0.15)
JITTER_ONLY = PoseErrors(0.0, 0.02, 0.0, 0.1, 0.0, 0.1)  # per-image noise of a good RTK log


def random_defects(scene: Scene, rng: np.random.Generator, counts=None) -> list[Defect]:
    """Defects at random modules: hot cells, substrings, offline modules and one offline string."""
    counts = counts or {"hot_cell": 16, "hot_cells": 6, "substring": 8, "offline": 6, "string": 1}
    mods = list(scene.modules())
    taken: set[str] = set()
    out: list[Defect] = []

    def free(m: Module) -> bool:  # keep defects apart so each module has one truth
        return all(
            f"R{m.row.index + 1:02d}-C{m.column + d:03d}-{lv}" not in taken
            for d in (-1, 0, 1)
            for lv in (m.level - 1, m.level, m.level + 1)
        )

    def pick() -> Module:
        while True:
            m = mods[rng.integers(len(mods))]
            if 2 <= m.column <= m.row.modules_along - 1 and free(m):
                taken.add(m.id)
                return m

    for _ in range(counts["hot_cell"]):
        m = pick()
        n_s, n_t = m.grid
        cell = (int(rng.integers(n_t)), int(rng.integers(n_s)))
        out.append(Defect(m.id, "hot_cell", float(rng.uniform(6, 25)), [cell]))
    for _ in range(counts["hot_cells"]):
        m = pick()
        n_s, n_t = m.grid
        r, c = int(rng.integers(1, n_t - 1)), int(rng.integers(0, n_s - 1))
        cells = [(r, c), (r + 1, c + 1)] if rng.random() < 0.5 else [(r, c), (r - 1, c), (r, c + 1)]
        out.append(Defect(m.id, "hot_cells", float(rng.uniform(6, 20)), cells))
    for _ in range(counts["substring"]):
        m = pick()
        out.append(Defect(m.id, "substring", float(rng.uniform(3.5, 8)), [], int(rng.integers(3))))
    for _ in range(counts["offline"]):
        m = pick()
        out.append(Defect(m.id, "offline", float(rng.uniform(3, 6))))
    for _ in range(counts["string"]):  # 8 neighbouring modules of one level offline together
        m = pick()
        start = min(m.column, m.row.modules_along - 8)
        dt = float(rng.uniform(3, 5))
        for c in range(start, start + 8):
            mid = f"R{m.row.index + 1:02d}-C{c:03d}-{m.level}"
            taken.add(mid)
            out = [d for d in out if d.module_id != mid]
            out.append(Defect(mid, "string", dt))
    return out


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


class ThermalTwin:
    def __init__(
        self,
        scene: Scene,
        defects: list[Defect],
        weather: Weather = Weather(),
        optics: Optics = Optics(),
        seed: int = 0,
        thermal: ModuleThermal = ModuleThermal(),
    ):
        self.scene, self.weather, self.optics, self.thermal = scene, weather, optics, thermal
        self.defects = {d.module_id: d for d in defects}
        self.rng = np.random.default_rng(seed)
        self.plant = Plant(scene.name, scene.frame, scene.rows, scene.timezone)
        lo = np.min([r.corners().min(axis=0) for r in scene.rows], axis=0) - 60
        hi = np.max([r.corners().max(axis=0) for r in scene.rows], axis=0) + 60
        self._g0, self._gstep = lo[:2], 0.5
        n = np.ceil((hi[:2] - lo[:2]) / self._gstep).astype(int) + 1
        field_ = ndimage.gaussian_filter(
            np.random.default_rng(seed + 1).normal(size=(n[1], n[0])), 4
        )
        self._ground = 2.0 * field_ / field_.std()  # +-2 K texture fixed to the ground
        self._cells: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ physics
    def conditions(self, time: pd.Timestamp) -> dict:
        cs = clear_sky_poa(self.plant, pd.DatetimeIndex([time]))
        poa = float(cs.poa.iloc[0])
        t_mod = float(pvlib.temperature.faiman(poa, self.weather.air_c, self.weather.wind_ms))
        sun = sun_vector(cs.elevation.iloc[0], cs.azimuth.iloc[0])
        return {
            "poa": poa,
            "t_module": t_mod,
            "sun": sun,
            "t_ground_sun": self.weather.air_c + 8.0,
            "t_ground_shade": self.weather.air_c + 1.0,
        }

    def _cell_offsets(self, module: Module) -> np.ndarray:
        if module.id not in self._cells:
            n_s, n_t = module.grid
            rng = np.random.default_rng(_seed(module.id))
            self._cells[module.id] = rng.normal(0, 1.0, (n_t, n_s)) * self.thermal.cell_sigma_k
        return self._cells[module.id]

    def _defect_dt(self, d: Defect, ms, mt, module: Module) -> np.ndarray:
        cw, ch = module.cell_size
        x, y = ms - FRAME_M, mt - FRAME_M
        if d.kind in ("offline", "string"):
            return np.full_like(ms, d.delta_t)
        if d.kind == "soiling":  # dirt along the lower edge shades the bottom cells: they warm
            rng = np.random.default_rng(_seed(module.id + "/soiling"))
            height = 0.12 + 0.18 * rng.random()
            waves, phase = rng.uniform(1.5, 4.0), rng.uniform(0, 2 * np.pi)
            along = 0.6 + 0.4 * np.sin(2 * np.pi * waves * x / module.width + phase)
            return d.delta_t * along / (1 + np.exp((y - height) / 0.03))
        if d.kind == "substring":  # two cells wide, along the long side
            u, pitch = (x, 2 * cw) if module.portrait else (y, 2 * ch)
            c0 = d.substring * pitch
            rise, fall = np.exp(-(u - c0) / 0.02), np.exp((u - c0 - pitch) / 0.02)
            inside = 1 / (1 + rise) * (1 / (1 + fall))
            return d.delta_t * inside
        out = np.zeros_like(ms)
        for r, c in d.cells:  # a hot cell plus heat spreading into its neighbours
            xc, yc = (c + 0.5) * cw, (r + 0.5) * ch
            core = np.exp(-(((x - xc) / (0.55 * cw)) ** 4) - ((y - yc) / (0.55 * ch)) ** 4)
            tail = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * (1.2 * cw) ** 2))
            out = np.maximum(out, d.delta_t * (0.8 * core + 0.2 * tail))
        return out

    def _pattern(self, ms, mt, module: Module) -> np.ndarray:
        """Junction-box patch and edge cooling of a healthy module."""
        th = self.thermal
        out = np.zeros_like(ms)
        if th.jbox_k:  # behind the top of the module, centred across its width
            w, h = module.width, module.height
            cx, cy = (w / 2, h - 0.18) if module.portrait else (w - 0.18, h / 2)
            out += th.jbox_k * np.exp(-((ms - cx) ** 2 + (mt - cy) ** 2) / (2 * 0.08**2))
        if th.edge_k:
            d = np.minimum.reduce(
                [
                    ms - FRAME_M,
                    module.width - FRAME_M - ms,
                    mt - FRAME_M,
                    module.height - FRAME_M - mt,
                ]
            )
            out += th.edge_k * np.exp(-np.maximum(d, 0) / 0.04)
        return out

    def _ground_texture(self, xy: np.ndarray) -> np.ndarray:
        g = (xy - self._g0) / self._gstep
        return ndimage.map_coordinates(self._ground, [g[:, 1], g[:, 0]], order=1, mode="nearest")

    def _shaded(self, P: np.ndarray, sun: np.ndarray) -> np.ndarray:
        shade = np.zeros(len(P), dtype=bool)
        for r in self.scene.rows:
            n = r.normal
            denom = sun @ n
            if abs(denom) < 1e-9:
                continue
            t = ((r.center - P) @ n) / denom
            X = P + t[:, None] * sun - r.center
            shade |= (
                (t > 0)
                & (np.abs(X @ r.along) <= r.length / 2)
                & (np.abs(X @ r.upslope) <= r.width / 2)
            )
        return shade

    # ------------------------------------------------------------------ render
    def render(
        self, pose: Pose, time: pd.Timestamp | None = None, speed_mps: float = 0.0
    ) -> np.ndarray:
        """Apparent temperature (deg C) seen from ``pose`` at ``time``."""
        cam, o = self.scene.camera, self.optics
        time = pose.time if time is None else time
        cond = self.conditions(time)
        u, v = pixel_grid(cam)
        d = rays(u, v, pose, cam).reshape(-1, 3)
        C = pose.position
        best_t = np.full(len(d), np.inf)
        hit_row = np.full(len(d), -1)
        S = np.zeros(len(d))
        T = np.zeros(len(d))
        proj: dict = {}  # ray directions projected once per row orientation (usually one)
        for k, r in enumerate(self.scene.rows):
            n = r.normal
            key = (*np.round(n, 12), *np.round(r.along, 12))
            if key not in proj:
                proj[key] = (d @ n, d @ r.along, d @ r.upslope)
            dn, da, du = proj[key]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((r.center - C) @ n) / dn
            c0 = C - r.center
            s = c0 @ r.along + t * da
            tt = c0 @ r.upslope + t * du
            hit = (t > 0) & (t < best_t) & (np.abs(s) <= r.length / 2) & (np.abs(tt) <= r.width / 2)
            best_t[hit], hit_row[hit], S[hit], T[hit] = t[hit], k, s[hit], tt[hit]

        img = np.empty(len(d))
        ground = hit_row < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = -C[2] / d[:, 2]
        G = C + tg[:, None] * d
        sky = ground & ~(tg > 0)
        gnd = ground & (tg > 0)
        shade = self._shaded(G[gnd], cond["sun"])
        img[gnd] = np.where(
            shade, cond["t_ground_shade"], cond["t_ground_sun"]
        ) + self._ground_texture(G[gnd, :2])
        img[sky] = self.weather.air_c - 20

        for k, r in enumerate(self.scene.rows):
            sel = np.flatnonzero(hit_row == k)
            if sel.size == 0:
                continue
            w, h = r.length / r.modules_along, r.width / r.modules_across
            s0, t0 = S[sel] + r.length / 2, T[sel] + r.width / 2
            col = np.minimum((s0 // w).astype(int), r.modules_along - 1)
            lvl = np.minimum((t0 // h).astype(int), r.modules_across - 1)
            ms, mt = s0 - col * w, t0 - lvl * h
            temp = cond["t_module"] + 1.0 * t0 / r.width  # warmer towards the top
            n = r.normal
            dn = d[sel] @ n
            front = dn < 0
            view = np.degrees(np.arccos(np.clip(-dn, -1, 1)))
            cj, ci = Module(r, 1, 1).cell_index(ms, mt)  # all modules of a row are alike
            for key in np.unique(col * 100 + lvl):
                m = (col * 100 + lvl) == key
                module = Module(r, int(key // 100 + 1), int(key % 100 + 1))
                temp[m] += self._cell_offsets(module)[cj[m], ci[m]]
                temp[m] += self._pattern(ms[m], mt[m], module)
                if module.id in self.defects:
                    temp[m] += self._defect_dt(self.defects[module.id], ms[m], mt[m], module)
            on_frame = (ms < FRAME_M) | (ms > w - FRAME_M) | (mt < FRAME_M) | (mt > h - FRAME_M)
            temp = np.where(on_frame, cond["t_module"] + o.frame_k, temp)
            temp -= 1.5 * np.clip(view / 60, 0, None) ** 4  # emissivity falls at grazing angles
            mirrored = d[sel] - 2 * dn[:, None] * n  # where the line of sight bounces to
            glare = np.degrees(np.arccos(np.clip(mirrored @ cond["sun"], -1, 1)))
            temp += np.where(
                on_frame, 0, o.reflection_k * np.exp(-0.5 * (glare / o.reflection_deg) ** 2)
            )
            img[sel] = np.where(
                front, temp, cond["t_module"] - 2.0
            )  # back sheet if seen from behind

        img = img.reshape(cam.height_px, cam.width_px)
        img = ndimage.gaussian_filter(img, o.blur_px)
        if speed_mps > 0:  # motion blur along the flight direction (image x)
            gsd = cam.gsd(
                float(np.median(best_t[np.isfinite(best_t)])) if np.isfinite(best_t).any() else 20
            )
            n_px = speed_mps * cam.time_constant_ms * 1e-3 / gsd
            if n_px > 0.3:
                img = ndimage.uniform_filter1d(img, max(1, int(round(n_px))), axis=1)
        rr = ((u - cam.width_px / 2) ** 2 + (v - cam.height_px / 2) ** 2) / (
            (cam.width_px / 2) ** 2 + (cam.height_px / 2) ** 2
        )
        img = img - o.vignetting_k * rr + self.rng.normal(0, o.netd_k, img.shape)
        return img.astype(np.float32)


def flown_poses(
    scene: Scene, rng: np.random.Generator, track_m: float = 0.03, heading_deg: float = 0.2
) -> list[Pose]:
    """True poses: the plan plus small tracking errors (ArduPilot SITL: ~3 cm, ~0.2 deg)."""
    out = []
    for p in scene.poses:
        dxy = rng.normal(0, track_m, 2)
        out.append(p.moved((dxy[0], dxy[1], rng.normal(0, track_m)), rng.normal(0, heading_deg)))
    return out


def logged_poses(true: list[Pose], errors: PoseErrors, rng: np.random.Generator) -> list[Pose]:
    """What the drone records: true pose + per-flight bias + per-image jitter."""
    e = errors
    bias = np.append(rng.normal(0, e.position_bias_m, 2), rng.normal(0, 1.5 * e.position_bias_m))
    bias_yaw, bias_pitch = rng.normal(0, e.yaw_bias_deg), rng.normal(0, e.pitch_bias_deg)
    out = []
    for p in true:
        jit = np.append(
            rng.normal(0, e.position_jitter_m, 2), rng.normal(0, 1.5 * e.position_jitter_m)
        )
        out.append(
            p.moved(
                bias + jit,
                bias_yaw + rng.normal(0, e.yaw_jitter_deg),
                bias_pitch + rng.normal(0, e.pitch_jitter_deg),
            )
        )
    return out


def fixed_bias_poses(
    true: list[Pose],
    rng: np.random.Generator,
    horizontal_m: float = 0.0,
    vertical_m: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    jitter: PoseErrors = JITTER_ONLY,
) -> list[Pose]:
    """Logged poses with a bias of exactly the given size (random direction and signs) plus
    per-image jitter, for sweeps over the size of the error."""
    a = rng.uniform(0, 2 * np.pi)
    bias = np.array(
        [horizontal_m * np.cos(a), horizontal_m * np.sin(a), vertical_m * rng.choice([-1, 1])]
    )
    yaw, pitch = yaw_deg * rng.choice([-1, 1]), pitch_deg * rng.choice([-1, 1])
    j = jitter
    out = []
    for p in true:
        d = np.append(
            rng.normal(0, j.position_jitter_m, 2), rng.normal(0, 1.5 * j.position_jitter_m)
        )
        out.append(
            p.moved(
                bias + d,
                yaw + rng.normal(0, j.yaw_jitter_deg),
                pitch + rng.normal(0, j.pitch_jitter_deg),
            )
        )
    return out


def _pose_dict(scene: Scene, p: Pose, file: str | None = None) -> dict:
    lat, lon = scene.frame.to_geodetic(p.position[0], p.position[1])
    out = {
        "line": p.line,
        "seq": p.seq,
        "time": p.time.isoformat(),
        "enu": [round(float(x), 4) for x in p.position],
        "lat": float(lat),
        "lon": float(lon),
        "alt_agl": round(float(p.position[2]), 4),
        "yaw_deg": round(p.yaw_deg, 4),
        "gimbal_pitch_deg": round(p.pitch_deg, 4),
    }
    if file:
        out = {"file": file, **out}
    return out


def write_survey(
    out: Path,
    scene: Scene,
    twin: ThermalTwin,
    logged: list[Pose],
    true: list[Pose],
    speeds: dict[int, float] | None = None,
    indices: list[int] | None = None,
    progress=None,
) -> Path:
    """Render frames (all, or ``indices``) to 16-bit radiometric TIFFs in centikelvin, the way
    thermal cameras store raw data, plus frames.json (logged poses) and truth.json."""
    out = Path(out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    indices = list(range(len(true))) if indices is None else list(indices)
    frames = []
    for n, i in enumerate(indices):
        name = f"frames/{i:05d}.tiff"
        img = twin.render(true[i], speed_mps=(speeds or {}).get(true[i].line, 0.0))
        ck = np.clip(np.round((img.astype(np.float64) + 273.15) * 100), 0, 65535).astype(np.uint16)
        Image.fromarray(ck).save(out / name, compression="tiff_adobe_deflate")
        frames.append(_pose_dict(scene, logged[i], name))
        if progress:
            progress(n + 1, len(indices))
    (out / "frames.json").write_text(
        json.dumps(
            {"format": "pvdefect-frames/1", "units": "centikelvin", "frames": frames}, indent=1
        )
    )
    truth = {
        "defects": [asdict(d) for d in twin.defects.values()],
        "true_poses": [_pose_dict(scene, true[i]) for i in indices],
        "weather": asdict(twin.weather),
        "optics": asdict(twin.optics),
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return out
