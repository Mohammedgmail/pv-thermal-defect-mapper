"""Score the mapper against the digital twin's ground truth.

A finding on a module that has a defect is a true positive. A finding anywhere else is a
false positive, including one on the neighbour of a defective module (the usual result of
a pose error). Type, cell and temperature accuracy are measured on true positives.
"""

from __future__ import annotations

import tempfile
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
    col = int(c[1:])
    out = {f"{r}-C{col + d:03d}-{lv}" for d in (-1, 0, 1) for lv in ("1", "2")}
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


class Survey:
    """A flight rendered once (frames kept on disk), mapped as often as needed."""

    def __init__(
        self,
        scene: Scene,
        defects: list[Defect],
        seed: int = 0,
        speeds: dict[int, float] | None = None,
        frames: list[int] | None = None,
        twin_kwargs: dict | None = None,
    ):
        self.scene, self.defects, self.seed = scene, defects, seed
        rng = np.random.default_rng(seed)
        self.true = flown_poses(scene, rng)
        self.index = list(range(len(scene.poses))) if frames is None else list(frames)
        self.twin = ThermalTwin(scene, defects, seed=seed, **(twin_kwargs or {}))
        cam = scene.camera
        self._dir = tempfile.TemporaryDirectory(prefix="pvdefect_")
        path = Path(self._dir.name) / "frames.f32"
        self.images = np.memmap(
            path, np.float32, "w+", shape=(len(self.index), cam.height_px, cam.width_px)
        )
        for k, i in enumerate(self.index):
            p = self.true[i]
            self.images[k] = self.twin.render(p, speed_mps=(speeds or {}).get(p.line, 0.0))
        self.images.flush()

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
