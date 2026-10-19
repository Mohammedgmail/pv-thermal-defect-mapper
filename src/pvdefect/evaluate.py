"""Score the mapper against the digital twin's ground truth.

A finding on a module that has a defect is a true positive. A finding anywhere else is a
false positive, including one on the neighbour of a defective module (the usual result of
a pose error). Type, cell and temperature accuracy are measured on true positives.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .mapper import Finding, Mapper, Settings
from .scene import Pose, Scene
from .simulate import Defect, PoseErrors, ThermalTwin, flown_poses, logged_poses


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    kind_accuracy: float
    cell_accuracy: float
    delta_t_mae: float
    next_door: int  # false positives right next to a missed defect (pose error signature)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _neighbours(mid: str) -> set[str]:
    r, c, lvl = mid.split("-")
    col, level = int(c[1:]), int(lvl)
    out = {f"{r}-C{col + d:03d}-{lv}" for d in (-1, 0, 1) for lv in (level - 1, level, level + 1)}
    out.discard(mid)
    return out


def score(findings: list[Finding], truth: list[Defect]) -> Score:
    t = {d.module_id: d for d in truth}
    f = {x.module_id: x for x in findings}
    tp_ids = [m for m in f if m in t]
    fp_ids = [m for m in f if m not in t]
    fn_ids = [m for m in t if m not in f]
    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    kinds = [f[m].kind == t[m].kind for m in tp_ids]
    cells = [
        tuple(f[m].cells[0]) == tuple(t[m].cells[0])
        for m in tp_ids
        if t[m].kind == "hot_cell" and f[m].kind == "hot_cell" and f[m].cells
    ]
    dts = [abs(f[m].delta_t - t[m].delta_t) for m in tp_ids]
    missed = set(fn_ids)
    next_door = sum(1 for m in fp_ids if _neighbours(m) & missed)
    return Score(
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
        float(np.mean(kinds)) if kinds else float("nan"),
        float(np.mean(cells)) if cells else float("nan"),
        float(np.mean(dts)) if dts else float("nan"),
        next_door,
    )


_RENDER: dict = {}  # shared with forked render workers


def _render_some(job) -> None:
    path, shape, items = job
    twin, true, speeds = _RENDER["twin"], _RENDER["true"], _RENDER["speeds"]
    out = np.memmap(path, np.float32, "r+", shape=shape)
    for k, i in items:
        # sensor noise drawn per frame, so it does not depend on which worker renders it
        twin.rng = np.random.default_rng([_RENDER["seed"], i])
        out[k] = twin.render(true[i], speed_mps=speeds.get(true[i].line, 0.0))
    out.flush()


class Survey:
    """A flight rendered once (frames kept on disk), mapped as often as needed.

    ``workers`` renders in that many forked processes. ``cache`` is a folder where the
    rendered frames are kept between runs, under a key made of everything that changes them
    (plan, poses, defects, seed, speeds, twin settings and the renderer's own source code).
    """

    def __init__(
        self,
        scene: Scene,
        defects: list[Defect],
        seed: int = 0,
        speeds: dict[int, float] | None = None,
        frames: list[int] | None = None,
        twin_kwargs: dict | None = None,
        workers: int = 1,
        cache: Path | None = None,
    ):
        self.scene, self.defects, self.seed = scene, defects, seed
        rng = np.random.default_rng(seed)
        self.true = flown_poses(scene, rng)
        self.index = list(range(len(scene.poses))) if frames is None else list(frames)
        self.twin = ThermalTwin(scene, defects, seed=seed, **(twin_kwargs or {}))
        cam = scene.camera
        shape = (len(self.index), cam.height_px, cam.width_px)
        speeds = speeds or {}
        if cache is not None:
            from . import simulate

            key = hashlib.sha256(
                repr(
                    (
                        [(p.position.round(6).tolist(), p.yaw_deg, p.pitch_deg, str(p.time))
                         for p in scene.poses],
                        [(r.center.tolist(), r.length, r.width, r.tilt, r.azimuth,
                          r.modules_along, r.modules_across) for r in scene.rows],
                        scene.frame, scene.camera, defects, seed, sorted(speeds.items()),
                        self.index, sorted((twin_kwargs or {}).items()),
                        Path(simulate.__file__).read_bytes(),
                    )
                ).encode()
            ).hexdigest()[:20]  # fmt: skip
            cache = Path(cache)
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / f"survey_{key}.f32"
            if path.exists() and path.stat().st_size == 4 * int(np.prod(shape)):
                self.images = np.memmap(path, np.float32, "r", shape=shape)
                return
            tmp = path.with_suffix(".part")
        else:
            self._dir = tempfile.TemporaryDirectory(prefix="pvdefect_")
            tmp = path = Path(self._dir.name) / "frames.f32"
        np.memmap(tmp, np.float32, "w+", shape=shape).flush()
        items = list(enumerate(self.index))
        _RENDER.update(twin=self.twin, true=self.true, speeds=speeds, seed=seed)
        if workers > 1:
            chunks = [(tmp, shape, items[w::workers]) for w in range(workers)]
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
                list(ex.map(_render_some, chunks))
        else:
            _render_some((tmp, shape, items))
        if tmp != path:
            tmp.rename(path)
        self.images = np.memmap(path, np.float32, "r", shape=shape)

    def logged(self, errors: PoseErrors, seed: int = 0) -> list[Pose]:
        poses = logged_poses(self.true, errors, np.random.default_rng(10_000 + seed))
        return [poses[i] for i in self.index]

    def frames(self, poses: list[Pose]):
        return [(self.images[k], poses[k]) for k in range(len(self.index))]

    def map(
        self,
        errors: PoseErrors,
        settings: Settings = Settings(),
        seed: int = 0,
        refine: bool = False,
    ) -> tuple[list[Finding], Score]:
        poses = self.logged(errors, seed)
        mapper = Mapper(self.scene, settings)
        frames = self.frames(poses)
        if refine:
            from .refine import refine_poses

            poses, self.refine_report = refine_poses(self.scene, frames)
            frames = self.frames(poses)
        findings, _ = mapper.run(frames)
        return findings, score(findings, self.defects)
