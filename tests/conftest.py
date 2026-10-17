from pathlib import Path

import numpy as np
import pytest

from pvdefect.scene import Scene
from pvdefect.simulate import ThermalTwin, flown_poses, random_defects

ROOT = Path(__file__).resolve().parents[1]
NOON = ROOT / "examples" / "june-noon" / "mission.json"
MORNING = ROOT / "examples" / "june-morning" / "mission.json"
LINE = 3  # flight line used for the rendered tests (row 4)


@pytest.fixture(scope="session")
def scene() -> Scene:
    return Scene.from_mission(NOON)


@pytest.fixture(scope="session")
def defects(scene):
    return random_defects(scene, np.random.default_rng(7))


@pytest.fixture(scope="session")
def rendered_pass(scene, defects):
    """One pass of the noon plan rendered from the true (flown) poses: (indices, images, true)."""
    true = flown_poses(scene, np.random.default_rng(7))
    twin = ThermalTwin(scene, defects, seed=7)
    speed = next(ln["speed_mps"] for ln in scene.extras["mission"]["lines"] if ln["index"] == LINE)
    idx = [i for i, p in enumerate(true) if p.line == LINE]
    imgs = [twin.render(true[i], speed_mps=speed) for i in idx]
    return idx, imgs, true
