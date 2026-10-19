import numpy as np
import pandas as pd

from pvdefect import reinspect as RI
from pvdefect.camera import project
from pvdefect.mapper import Finding, View


def test_candidates_follow_the_rules():
    views = [
        View("R01-C010-1", 0, "cell", 8.0, [(1, 1)], 1),  # flagged once ...
        View("R01-C010-1", 1, "none", reflections=1),  # ... and dropped as a reflection once
        View("R01-C020-1", 2, "cell", 8.0, [(1, 1)], 1),  # one view only
        View("R01-C030-1", 3, "cell", 8.0, [(2, 2)], 1),
        View("R01-C030-1", 4, "none"),
        View("R01-C040-1", 5, "none", reflections=2),  # reflection on a healthy module
    ]
    found = [Finding("R01-C030-1", "hot_cell", 8.0, 1, 2, [(2, 2)])]
    got = {(c.module_id, c.reason) for c in RI.candidates(found, views)}
    assert got == {
        ("R01-C010-1", "masked by glare"),
        ("R01-C020-1", "too few views"),
        ("R01-C030-1", "views disagree"),
    }


def test_viewpoint_meets_resolution_glare_and_framing(scene):
    m = scene.module("R08-C050-2")
    sun = RI.sun_at(scene, scene.poses[0].time)
    vp = RI.viewpoint(scene, m, sun, gsd_m=0.02, min_glare_deg=15)
    assert vp is not None
    assert vp.gsd_cm <= 2.0 + 1e-6 and vp.view_deg <= 40 and vp.glare_deg >= 15
    from pvdefect.scene import Pose

    u, v, depth = project(m.corners(), Pose(vp.position, vp.yaw_deg, vp.pitch_deg), scene.camera)
    assert (depth > 0).all() and u.min() > 0 and u.max() < scene.camera.width_px
    assert v.min() > 0 and v.max() < scene.camera.height_px


def test_plan_orders_the_tour_and_writes_an_ardupilot_mission(scene, tmp_path):
    todo = [
        RI.Candidate(mid, "too few views") for mid in ("R02-C010-1", "R15-C090-2", "R03-C012-1")
    ]
    home = np.array([60.0, -95.0, 0.0])
    p = RI.plan(scene, todo, scene.poses[0].time + pd.Timedelta(minutes=30), home)
    assert len(p.viewpoints) == 3 and not p.unreachable
    # the two neighbouring targets are visited one after the other
    ids = [v.module_id for v in p.viewpoints]
    assert abs(ids.index("R02-C010-1") - ids.index("R03-C012-1")) == 1
    assert 60 < p.flight_s < 600
    path = tmp_path / "reinspect.waypoints"
    RI.write_waypoints(path, scene, p, home)
    lines = path.read_text().splitlines()
    assert lines[0] == "QGC WPL 110"
    commands = [int(x.split("\t")[3]) for x in lines[1:]]
    assert commands.count(2000) == 3 and commands[-1] == 20  # three images, then return home
