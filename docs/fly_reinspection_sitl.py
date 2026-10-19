"""Fly a re-inspection mission (from docs/run_experiments.py reinspection) in ArduPilot SITL.

Uses the SITL helpers of pv-drone-inspection-planner (sitl/fly_sitl.py), so it needs a checkout
of that repo and an ArduPilot tree with ArduCopter SITL built:

    python docs/fly_reinspection_sitl.py --planner ~/GitHub/pv-drone-inspection-planner \\
        --ardupilot ~/ardupilot --mission examples/june-morning/mission.json \\
        --waypoints docs/results/reinspection/morning_plan_at_1300_step2.waypoints

Checks that every viewpoint is reached, that one image is taken at each, and how long the
flight takes compared with the planned time. Writes <waypoints>.sitl.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--planner", required=True, type=Path, help="pv-drone-inspection-planner checkout"
    )
    ap.add_argument("--ardupilot", required=True, type=Path)
    ap.add_argument("--mission", required=True, type=Path, help="the survey's mission.json")
    ap.add_argument("--waypoints", required=True, type=Path)
    ap.add_argument("--speedup", type=float, default=5.0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.planner.expanduser().resolve() / "sitl"))
    from fly_sitl import arm_and_start, connect, enu, request_streams, start_sitl, upload
    from pymavlink import mavwp

    plan = json.loads(a.mission.read_text())
    lat0, lon0 = plan["plant"]["latitude"], plan["plant"]["longitude"]
    home = plan["home_lla"]
    loader = mavwp.MAVWPLoader()
    loader.load(str(a.waypoints))
    wps = [loader.wp(i) for i in range(1, loader.count()) if loader.wp(i).command == 16]
    targets = np.array([[*enu(w.x, w.y, lat0, lon0), w.z] for w in wps])
    params = {"WP_YAW_BEHAVIOR": 0, "AUTO_OPTIONS": 3, "CAM1_TYPE": 1, "SERVO9_FUNCTION": 10}
    work = Path(tempfile.mkdtemp(prefix="pvdefect_sitl_"))
    sitl = start_sitl(
        a.ardupilot, (home[0], home[1], plan["plant"]["altitude"]), a.speedup, params, work
    )
    track, shots = [], []
    try:
        m = connect()
        request_streams(m)
        upload(m, loader)
        arm_and_start(m)
        t_takeoff = t_land = None
        armed_seen = False
        while True:
            msg = m.recv_match(blocking=True, timeout=30)
            if msg is None:
                raise RuntimeError("SITL stopped sending data")
            kind = msg.get_type()
            if kind == "GLOBAL_POSITION_INT":
                t, alt = msg.time_boot_ms / 1e3, msg.relative_alt / 1e3
                track.append((t, *enu(msg.lat / 1e7, msg.lon / 1e7, lat0, lon0), alt))
                if t_takeoff is None and alt > 0.2:
                    t_takeoff = t
            elif kind == "CAMERA_FEEDBACK":
                shots.append((*enu(msg.lat / 1e7, msg.lng / 1e7, lat0, lon0), msg.alt_rel))
            elif kind == "HEARTBEAT" and msg.get_srcSystem() == m.target_system:
                armed = bool(msg.base_mode & 128)
                armed_seen |= armed
                if armed_seen and not armed:
                    t_land = track[-1][0] if track else None
                    break
    finally:
        sitl.terminate()
        time.sleep(1)
    tr = np.array(track)
    reached = [float(np.min(np.linalg.norm(tr[:, 1:4] - p, axis=1))) for p in targets]
    shot_err = [
        float(np.min(np.linalg.norm(np.array(shots)[:, :3] - p, axis=1))) if shots else float("nan")
        for p in targets
    ]
    out = {
        "viewpoints": len(targets),
        "viewpoints_reached_2m": int(sum(d < 2.0 for d in reached)),
        "images": len(shots),
        "image_to_viewpoint_m_max": round(max(shot_err), 2) if shots else None,
        "flight_minutes_sitl": round((t_land - t_takeoff) / 60, 2)
        if t_land and t_takeoff
        else None,
    }
    print(json.dumps(out, indent=1))
    out["targets_enu"] = targets.round(2).tolist()
    out["shots_enu"] = np.array(shots).round(2).tolist() if shots else []
    out["track_enu"] = tr[:: max(1, len(tr) // 400), 1:4].round(2).tolist()
    a.waypoints.with_suffix(".sitl.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
