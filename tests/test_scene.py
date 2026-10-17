import numpy as np
import pandas as pd

from pvdefect.camera import project, rays


def test_mission_is_read_with_its_geometry(scene):
    m = scene.extras["mission"]
    assert len(scene.rows) == len(m["plant"]["rows"]) == 20
    assert len(scene.poses) == len(m["images"]) == 500
    assert scene.n_modules == sum(r.modules_along * r.modules_across for r in scene.rows)
    mod = scene.module("R04-C022-1")
    assert mod.id == "R04-C022-1"
    assert np.isclose(mod.width * mod.row.modules_along, mod.row.length)
    # the four corners lie in the row plane, inside the row
    c = mod.corners() - mod.row.center
    assert np.allclose(c @ mod.row.normal, 0, atol=1e-9)
    assert (np.abs(c @ mod.row.along) <= mod.row.length / 2 + 1e-9).all()


def test_modules_tile_the_row_without_gaps(scene):
    row = scene.rows[0]
    first, last = scene.module("R01-C001-1"), scene.module(f"R01-C{row.modules_along:03d}-2")
    assert np.allclose(first.corners()[0], row.point(-row.length / 2, -row.width / 2))
    assert np.allclose(last.corners()[2], row.point(row.length / 2, row.width / 2))


def test_pixel_ray_round_trip(scene):
    pose, cam = scene.poses[40], scene.camera
    u, v = np.array([10.0, 319.5, 600.0]), np.array([20.0, 255.5, 500.0])
    d = rays(u, v, pose, cam)
    P = pose.position + 25.0 * d
    uu, vv, depth = project(P, pose, cam)
    assert np.allclose(uu, u, atol=1e-6) and np.allclose(vv, v, atol=1e-6)
    assert (depth > 0).all()


def test_image_centre_looks_at_the_planned_row(scene):
    """The planner aims each image at the middle of its rows: the centre ray hits that row."""
    for pose in scene.poses[::37]:
        line = scene.extras["mission"]["lines"][pose.line]
        row = scene.rows[line["rows"][0] - 1]
        d = rays(np.array([319.5]), np.array([255.5]), pose, scene.camera)[0]
        t = ((row.center - pose.position) @ row.normal) / (d @ row.normal)
        hit = pose.position + t * d - row.center
        assert abs(hit @ row.upslope) < row.width / 2


def test_flown_at_keeps_the_spacing_between_images(scene):
    later = scene.flown_at("15:30")
    assert later.poses[0].time.strftime("%H:%M") == "15:30"
    gaps = np.diff([p.time.value for p in scene.poses])
    assert np.array_equal(np.diff([p.time.value for p in later.poses]), gaps)
    assert later.poses[0].time.tz is not None and isinstance(later.poses[0].time, pd.Timestamp)
