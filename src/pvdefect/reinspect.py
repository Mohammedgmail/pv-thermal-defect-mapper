"""Close the loop: plan a short re-inspection flight for what the survey left uncertain.

Candidates, as (module, reason):

* "masked by glare": no finding, but one view flagged a hot spot and another had a hot spot
  dropped by the glare guard. The defect may sit under the sun's reflection.
* "views disagree": a finding that only some of its views support.
* "too few views": a module with a flagged view but fewer than two views in total.

Each candidate gets a viewpoint that:

* sees the module closer than the survey did (finer resolution);
* stays within the viewing-angle limit;
* keeps the sun's reflection at least ``min_glare_deg`` away at the time of the re-flight;
* has no other row in the way.

The viewpoints are ordered by a nearest-neighbour tour improved with 2-opt. They are exported
as an ArduPilot mission: fly to each viewpoint, turn, set the gimbal and take one image. The
flight time comes from the planner's ArduCopter timing model.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pvmission.planner import Drone
from pvmission.plant import Plant, direction
from pvmission.sun import solar_position, sun_vector

from .camera import focal_px, project
from .mapper import Finding, View, _hidden
from .scene import Module, Pose, Scene

NAV_WAYPOINT, NAV_RTL, NAV_TAKEOFF = 16, 20, 22
CONDITION_YAW, DO_MOUNT_CONTROL, IMAGE_START_CAPTURE = 115, 205, 2000
FRAME_GLOBAL, FRAME_MISSION, FRAME_RELATIVE_ALT = 0, 2, 3


@dataclass
class Candidate:
    module_id: str
    reason: str


@dataclass
class Viewpoint:
    module_id: str
    reason: str
    position: np.ndarray  # ENU, z above ground
    yaw_deg: float
    pitch_deg: float
    view_deg: float  # from the module normal
    glare_deg: float  # sun reflection to line of sight
    gsd_cm: float  # on the module


def candidates(findings: list[Finding], views: list[View]) -> list[Candidate]:
    """Modules worth a second look (see the module docstring)."""
    by: dict[str, list[View]] = defaultdict(list)
    for v in views:
        by[v.module_id].append(v)
    found = {f.module_id: f for f in findings}
    out = []
    for mid, vs in sorted(by.items()):
        flagged = [v for v in vs if v.kind != "none"]
        if mid in found:
            if found[mid].views_flagged < found[mid].views_seen:
                out.append(Candidate(mid, "views disagree"))
        elif flagged and any(v.reflections > 0 for v in vs):
            out.append(Candidate(mid, "masked by glare"))
        elif flagged and len(vs) < 2:
            out.append(Candidate(mid, "too few views"))
    return out


def sun_at(scene: Scene, time: pd.Timestamp) -> np.ndarray:
    plant = Plant(scene.name, scene.frame, scene.rows, scene.timezone)
    sp = solar_position(plant, pd.DatetimeIndex([time]))
    return sun_vector(sp.apparent_elevation.iloc[0], sp.azimuth.iloc[0])


def viewpoint(
    scene: Scene,
    module: Module,
    sun: np.ndarray,
    reason: str = "",
    gsd_m: float = 0.02,
    max_view_deg: float = 40.0,
    min_glare_deg: float = 15.0,
    min_height_m: float = 5.0,
) -> Viewpoint | None:
    """The sharpest valid viewpoint of one module, or None if there is none.

    Resolution, viewing angle and the angle to the sun's reflection are checked at the four
    corners and the centre of the module (the worst of the five counts), like the survey
    planner checks every sample point of a pass."""
    cam = scene.camera
    f = focal_px(cam)
    row, c = module.row, module.center
    n = row.normal
    face_az = float(np.degrees(np.arctan2(n[0], n[1])))
    near = [r for r in scene.rows if r is not row and abs(r.index - row.index) <= 3]
    corners = module.corners()
    samples = np.vstack([corners, c[None, :]])

    def worst(pos):
        look = samples - pos
        dist = np.linalg.norm(look, axis=1)
        look /= dist[:, None]
        cos_view = np.clip(-look @ n, -1, 1)
        mirrored = look - 2 * (look @ n)[:, None] * n
        glare = np.degrees(np.arccos(np.clip(mirrored @ sun, -1, 1)))
        with np.errstate(divide="ignore"):
            gsd = dist / (f * cos_view)
        return float(np.degrees(np.arccos(cos_view.min()))), float(glare.min()), float(gsd.max())

    best = None
    for az in np.arange(-60.0, 60.1, 5.0):
        for el in np.arange(20.0, 85.1, 5.0):
            out = direction(face_az + az, el)  # from the module towards the camera
            view_c = float(np.degrees(np.arccos(np.clip(out @ n, -1, 1))))
            if view_c > max_view_deg:
                continue
            rng = gsd_m * f * np.cos(np.radians(view_c))
            for _ in range(5):  # come closer until the worst corner is sharp enough
                view, glare, gsd = worst(c + rng * out)
                if gsd <= gsd_m:
                    break
                rng *= 0.9995 * gsd_m / gsd
            pos = c + rng * out
            if pos[2] < min_height_m or view > max_view_deg or gsd > gsd_m:
                continue
            if sun[2] > 0 and glare < min_glare_deg:
                continue
            look = -out
            yaw = float(np.degrees(np.arctan2(look[0], look[1])))
            pitch = float(np.degrees(np.arcsin(np.clip(look[2], -1, 1))))
            pose = Pose(pos, yaw, pitch)
            u, v, depth = project(corners, pose, cam)
            margin = 20
            if (depth <= 0).any() or u.min() < margin or v.min() < margin:
                continue
            if u.max() > cam.width_px - margin or v.max() > cam.height_px - margin:
                continue
            if _hidden(corners, pos, near).any():
                continue
            key = (view, abs(az))
            if best is None or key < best[0]:
                best = (key, Viewpoint(module.id, reason, pos, yaw, pitch, view, glare, 100 * gsd))
    return None if best is None else best[1]


def tour(points: np.ndarray, start: np.ndarray) -> list[int]:
    """Visiting order: nearest neighbour from ``start``, then 2-opt."""
    left = list(range(len(points)))
    order, here = [], start
    while left:
        j = min(left, key=lambda k: np.linalg.norm(points[k] - here))
        order.append(j)
        left.remove(j)
        here = points[j]
    path = np.vstack([start, points[order], start])

    def length(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    improved = True
    while improved and len(order) > 3:
        improved = False
        for i in range(1, len(order)):
            for k in range(i + 1, len(order) + 1):
                cand = np.vstack([path[:i], path[i : k + 1][::-1], path[k + 1 :]])
                if length(cand) < length(path) - 1e-9:
                    path = cand
                    order = order[: i - 1] + order[i - 1 : k][::-1] + order[k:]
                    improved = True
    return order


@dataclass
class Plan:
    viewpoints: list[Viewpoint]  # in flight order
    unreachable: list[Candidate]
    flight_s: float
    start: pd.Timestamp
    times: list[pd.Timestamp]  # when each image is taken


def plan(
    scene: Scene,
    todo: list[Candidate],
    start: pd.Timestamp,
    home: np.ndarray,
    drone: Drone = Drone(),
    shot_s: float = 3.0,
    **kw,
) -> Plan:
    """Viewpoints for the candidates, in flight order, with the timing of each image."""
    sun = sun_at(scene, start)
    vps, missing = [], []
    for cnd in todo:
        vp = viewpoint(scene, scene.module(cnd.module_id), sun, cnd.reason, **kw)
        (vps.append(vp) if vp is not None else missing.append(cnd))
    if not vps:
        return Plan([], missing, 0.0, start, [])
    order = tour(np.array([v.position for v in vps]), np.array([*home[:2], 0.0]))
    vps = [vps[i] for i in order]
    t = drone.vertical_s(vps[0].position[2])  # take off
    here = np.array([*home[:2], vps[0].position[2]])
    times = []
    for v in vps:
        t += drone.travel_s(here, v.position) + drone.hold_s
        times.append(start + pd.Timedelta(seconds=t))
        t += shot_s
        here = v.position
    t += drone.home_and_land_s(here, np.array([*home[:2], 0.0]))
    return Plan(vps, missing, float(t), start, times)


def poses(p: Plan, first_line: int = 1000) -> list[Pose]:
    """Planned camera poses of the re-inspection images, one flight line each."""
    return [
        Pose(v.position.copy(), v.yaw_deg, v.pitch_deg, t, first_line + i, first_line + i)
        for i, (v, t) in enumerate(zip(p.viewpoints, p.times, strict=True))
    ]


def scene_with(scene: Scene, p: Plan, first_line: int = 1000) -> Scene:
    """A copy of ``scene`` whose mission also knows the re-inspection lines (one per image,
    covering the target module's row), so the mapper can use those images."""
    extras = copy.deepcopy(scene.extras)
    lines = extras.setdefault("mission", {}).setdefault("lines", [])
    for i, v in enumerate(p.viewpoints):
        row = int(v.module_id[1:3])
        lines.append({"index": first_line + i, "rows": [row], "speed_mps": 0.0})
    return Scene(
        scene.name, scene.frame, scene.timezone, scene.rows, scene.camera, scene.poses, extras
    )


def write_waypoints(
    path: Path, scene: Scene, p: Plan, home: np.ndarray, drone: Drone = Drone()
) -> None:
    """ArduPilot mission (QGC WPL 110): take off, then at each viewpoint turn, set the gimbal,
    stop, take one image; finally return to launch."""
    fr = scene.frame

    def lla(x):
        lat, lon = fr.to_geodetic(x[0], x[1])
        return float(lat), float(lon)

    hlat, hlon = lla(home)
    rows = [[0, 1, FRAME_GLOBAL, NAV_WAYPOINT, 0, 0, 0, 0, hlat, hlon, fr.altitude, 1]]

    def add(cmd, frame, p1=0.0, p2=0.0, p3=0.0, p4=0.0, x=0.0, y=0.0, z=0.0):
        rows.append([len(rows), 0, frame, cmd, p1, p2, p3, p4, x, y, z, 1])

    if p.viewpoints:
        add(
            NAV_TAKEOFF,
            FRAME_RELATIVE_ALT,
            x=hlat,
            y=hlon,
            z=round(float(p.viewpoints[0].position[2]), 2),
        )
    for v in p.viewpoints:
        lat, lon = lla(v.position)
        add(CONDITION_YAW, FRAME_MISSION, round(v.yaw_deg % 360, 2), 30, 0, 0)
        add(DO_MOUNT_CONTROL, FRAME_MISSION, round(v.pitch_deg, 2), 0, 0, 0, 0, 0, 2)
        add(
            NAV_WAYPOINT,
            FRAME_RELATIVE_ALT,
            drone.hold_s,
            0,
            0,
            0,
            lat,
            lon,
            round(float(v.position[2]), 2),
        )
        # one image (DO_DIGICAM_CONTROL does not trigger the camera in ArduCopter 4.7 SITL)
        add(IMAGE_START_CAPTURE, FRAME_MISSION, 0, 0, 1, 0)
    add(NAV_RTL, FRAME_MISSION)
    with open(path, "w") as f:
        f.write("QGC WPL 110\n")
        for r in rows:
            ints = [str(int(x)) for x in r[:4]]
            params = [
                f"{float(x):.8f}" if i in (4, 5) else f"{float(x):.6g}"
                for i, x in enumerate(r[4:11])
            ]
            f.write("\t".join([*ints, *params, str(int(r[11]))]) + "\n")
