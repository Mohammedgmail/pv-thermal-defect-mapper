import json

import numpy as np
import pytest

from pvdefect.mapper import Mapper, Settings, load_survey
from pvdefect.scene import Scene
from pvdefect.simulate import GNSS, RTK, ThermalTwin, flown_poses, logged_poses, write_survey

from .conftest import MORNING


def test_conditions_are_plausible_for_a_june_noon(scene, defects):
    c = ThermalTwin(scene, defects).conditions(scene.poses[0].time)
    assert 800 < c["poa"] < 1100  # W/m2, clear sky on the module plane
    assert 35 < c["t_module"] < 55  # deg C at 21 C air, 3 m/s wind
    assert c["sun"][2] > 0.8  # sun high in the sky


def test_defects_are_apart_and_cover_every_kind(defects):
    kinds = {d.kind for d in defects}
    assert kinds == {"hot_cell", "hot_cells", "substring", "offline", "string"}
    ids = [d.module_id for d in defects]
    assert len(ids) == len(set(ids))


def test_logged_poses_follow_the_error_model(scene):
    true = flown_poses(scene, np.random.default_rng(0))
    for errors, lo, hi in ((RTK, 0.0, 0.2), (GNSS, 0.3, 8.0)):
        logged = logged_poses(true, errors, np.random.default_rng(1))
        d = np.array([q.position - p.position for p, q in zip(true, logged, strict=True)])
        bias = np.linalg.norm(d.mean(axis=0)[:2])
        assert lo < bias < hi
        jitter = d[:, 0].std()
        assert jitter == pytest.approx(errors.position_jitter_m, rel=0.2)


def test_survey_round_trip_keeps_centikelvin(tmp_path, scene, defects):
    true = flown_poses(scene, np.random.default_rng(0))
    twin = ThermalTwin(scene, defects, seed=3)
    write_survey(tmp_path, scene, twin, true, true, indices=[80, 81])
    index = json.loads((tmp_path / "frames.json").read_text())
    assert index["units"] == "centikelvin" and len(index["frames"]) == 2
    frames = load_survey(tmp_path)
    again = ThermalTwin(scene, defects, seed=3).render(true[80])
    assert np.abs(frames[0][0] - again).max() <= 0.006
    assert np.allclose(frames[0][1].position, true[80].position, atol=1e-4)
    truth = json.loads((tmp_path / "truth.json").read_text())
    assert len(truth["defects"]) == len(defects)


def test_glare_guard_drops_the_reflection_of_the_sun(defects):
    """The morning plan flown at noon looks straight into the sun's reflection. Without the
    guard a single frame shows hot spots on healthy modules; with it, none."""
    scene = Scene.from_mission(MORNING).flown_at("13:00")
    pose = scene.poses[12]
    img = ThermalTwin(scene, defects, seed=1).render(pose)
    defect_ids = {d.module_id for d in defects}

    def clean_hot(guard):
        views = Mapper(scene, Settings(reflection_guard_deg=guard)).frame_views(0, img, pose)
        return [v for v in views if v.kind == "cell" and v.module_id not in defect_ids], views

    no_guard, _ = clean_hot(0.0)
    guarded, views = clean_hot(6.0)
    assert len(no_guard) >= 3
    assert guarded == []
    assert sum(v.reflections for v in views) >= len(no_guard)


def test_landscape_rows_render_and_map_to_the_right_cell():
    """A 4L landscape plant (Gujarat example): substrings run along the row and cells are
    counted 12 along the row by 6 up the slope."""
    from pvdefect.simulate import Defect

    from .conftest import ROOT

    scene = Scene.from_mission(ROOT / "examples" / "sites" / "gujarat" / "mission.json")
    m = scene.module("R03-C020-2")
    assert not m.portrait and m.grid == (12, 6)
    defects = [
        Defect("R03-C020-2", "hot_cell", 12.0, [(4, 9)]),
        Defect("R03-C030-3", "substring", 5.0, [], 2),
    ]
    true = flown_poses(scene, np.random.default_rng(0))
    twin = ThermalTwin(scene, defects, seed=1)
    frames = [(twin.render(p), p) for p in true if p.line == 2]
    found = {f.module_id: f for f in Mapper(scene).run(frames)[0]}
    assert found["R03-C020-2"].kind == "hot_cell" and found["R03-C020-2"].cells == [(4, 9)]
    assert found["R03-C030-3"].kind == "substring" and found["R03-C030-3"].substring == 2
