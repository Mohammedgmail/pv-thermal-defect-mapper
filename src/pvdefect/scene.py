"""The inspection scene from a pvmission plan: plant rows, modules, camera and image poses.

Everything is read from ``mission.json`` written by pv-drone-inspection-planner, so the
mapper uses exactly the geometry the flight was planned with. Coordinates are local
east-north-up metres around the plant reference point.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from pvmission.camera import Camera
from pvmission.plant import GeoFrame, Row, direction

CELLS_SHORT, CELLS_LONG = 6, 12  # 72-cell module: 6 cells across its short side, 12 along
CELLS_ACROSS, CELLS_ALONG = CELLS_SHORT, CELLS_LONG  # portrait: along the row x up the slope
SUBSTRINGS = 3  # bypass-diode substrings, each 2 cells wide, along the module's long side
FRAME_M = 0.035  # aluminium frame width


@dataclass(frozen=True)
class Pose:
    """Camera pose: ENU position (m, z above ground), heading and gimbal pitch (deg)."""

    position: np.ndarray
    yaw_deg: float
    pitch_deg: float
    time: pd.Timestamp | None = None
    line: int = -1
    seq: int = -1
    # per-row corrections from pose refinement: ((row index, (dx, dy, dz)), ...), applied when
    # this frame is used for the modules of that row (see for_row)
    row_shifts: tuple = ()

    def for_row(self, row_index: int) -> Pose:
        """The pose to use for the modules of one row: the row's own correction, if any."""
        for ri, d in self.row_shifts:
            if ri == row_index:
                return replace(self.moved(d), row_shifts=())
        return self

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors of the image: x to the right, y down, z along the optical axis."""
        z = direction(self.yaw_deg, self.pitch_deg)
        x = direction(self.yaw_deg + 90.0)
        return x, np.cross(z, x), z

    def moved(self, dpos=(0.0, 0.0, 0.0), dyaw=0.0, dpitch=0.0) -> Pose:
        return Pose(
            self.position + np.asarray(dpos, float),
            self.yaw_deg + dyaw,
            self.pitch_deg + dpitch,
            self.time,
            self.line,
            self.seq,
            self.row_shifts,
        )


@dataclass(frozen=True)
class Module:
    """One module: row, column (1-based along the row), level (1 = lower) and its corners."""

    row: Row
    column: int
    level: int

    @property
    def id(self) -> str:
        return f"R{self.row.index + 1:02d}-C{self.column:03d}-{self.level}"

    @property
    def width(self) -> float:  # along the row
        return self.row.length / self.row.modules_along

    @property
    def height(self) -> float:  # up the slope
        return self.row.width / self.row.modules_across

    @property
    def origin_st(self) -> tuple[float, float]:
        """(s, t) of the module's lower corner in its row's coordinates."""
        return (
            -self.row.length / 2 + (self.column - 1) * self.width,
            -self.row.width / 2 + (self.level - 1) * self.height,
        )

    @property
    def portrait(self) -> bool:
        """Long side up the slope (portrait) or along the row (landscape)."""
        return self.height >= self.width

    @property
    def grid(self) -> tuple[int, int]:
        """Number of cells along the row and up the slope: 6 x 12 in portrait, 12 x 6 in
        landscape."""
        return (CELLS_SHORT, CELLS_LONG) if self.portrait else (CELLS_LONG, CELLS_SHORT)

    @property
    def cell_size(self) -> tuple[float, float]:
        """Cell pitch (m) along the row and up the slope, inside the frame."""
        n_s, n_t = self.grid
        return (self.width - 2 * FRAME_M) / n_s, (self.height - 2 * FRAME_M) / n_t

    def cell_index(self, ms, mt):
        """(cell row from the bottom, cell column along the row) of module coordinates."""
        n_s, n_t = self.grid
        cw, ch = self.cell_size
        r = np.clip(((np.asarray(mt) - FRAME_M) // ch).astype(int), 0, n_t - 1)
        c = np.clip(((np.asarray(ms) - FRAME_M) // cw).astype(int), 0, n_s - 1)
        return r, c

    def substring_index(self, ms, mt):
        """Bypass-diode substring (0-2) of module coordinates: substrings are two cells wide
        and run along the long side, so they split the short side into thirds."""
        cw, ch = self.cell_size
        if self.portrait:
            x, pitch = np.asarray(ms) - FRAME_M, 2 * cw
        else:
            x, pitch = np.asarray(mt) - FRAME_M, 2 * ch
        return np.clip((x // pitch).astype(int), 0, SUBSTRINGS - 1)

    def point(self, ms, mt) -> np.ndarray:
        """World point at ``ms`` metres along and ``mt`` metres up from the lower corner."""
        s0, t0 = self.origin_st
        return self.row.point(s0 + np.asarray(ms, float), t0 + np.asarray(mt, float))

    def corners(self) -> np.ndarray:
        w, h = self.width, self.height
        return self.point(np.array([0, w, w, 0]), np.array([0, 0, h, h]))

    @property
    def center(self) -> np.ndarray:
        return self.point(self.width / 2, self.height / 2)


@dataclass
class Scene:
    name: str
    frame: GeoFrame
    timezone: str
    rows: list[Row]
    camera: Camera
    poses: list[Pose]
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_mission(cls, path_or_dict) -> Scene:
        m = path_or_dict
        if not isinstance(m, dict):
            m = json.loads(Path(path_or_dict).read_text())
        if m.get("format") != "pvmission/1":
            raise ValueError("not a pvmission/1 mission.json")
        p = m["plant"]
        rows = [
            Row(
                r["row"] - 1,
                np.array(r["center_enu"], float),
                r["length_m"],
                r["width_m"],
                r["tilt_deg"],
                r["azimuth_deg"],
                r["modules_along"],
                r["modules_across"],
            )
            for r in p["rows"]
        ]
        c = dict(m["camera"])
        c["pitch_range_deg"] = tuple(c["pitch_range_deg"])
        poses = [
            Pose(
                np.array(i["enu"], float),
                i["yaw_deg"],
                i["gimbal_pitch_deg"],
                pd.Timestamp(i["time"]),
                i["line"],
                i["seq"],
            )
            for i in m["images"]
        ]
        return cls(
            p["name"],
            GeoFrame(p["latitude"], p["longitude"], p["altitude"]),
            p["timezone"],
            rows,
            Camera(**c),
            poses,
            {"mission": m},
        )

    def flown_at(self, start: str) -> Scene:
        """The same plan flown at another local time of the same day ("HH:MM"): every image
        keeps its offset from the first one. Used to show what a plan does outside the time it
        was planned for."""
        t0 = self.poses[0].time
        hh, mm = (int(x) for x in start.split(":"))
        shift = t0.replace(hour=hh, minute=mm, second=0, microsecond=0, nanosecond=0) - t0
        poses = [dataclasses.replace(p, time=p.time + shift) for p in self.poses]
        return dataclasses.replace(self, poses=poses)

    def module(self, module_id: str) -> Module:
        r, c, lvl = module_id.split("-")
        return Module(self.rows[int(r[1:]) - 1], int(c[1:]), int(lvl))

    def modules(self):
        for r in self.rows:
            for c in range(1, r.modules_along + 1):
                for lvl in range(1, r.modules_across + 1):
                    yield Module(r, c, lvl)

    @property
    def n_modules(self) -> int:
        return sum(r.modules_along * r.modules_across for r in self.rows)
