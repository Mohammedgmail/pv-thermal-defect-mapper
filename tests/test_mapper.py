import numpy as np
import pytest

from pvdefect.evaluate import score
from pvdefect.mapper import Mapper, Settings, View, analyse_crop, crop_grid
from pvdefect.scene import CELLS_ACROSS, CELLS_ALONG, FRAME_M
from pvdefect.simulate import Defect

RES = 0.03


@pytest.fixture
def module(scene):
    return scene.module("R04-C050-1")


def synthetic_crop(m, rng, cell=None, band=None, dt=8.0, noise=0.05):
    """Healthy module at 40 C with cool frames, plus a hot cell (row, col) or a warm third."""
    ms, mt = crop_grid(m, RES)
    crop = 40.0 + rng.normal(0, noise, ms.shape)
    cw, ch = (m.width - 2 * FRAME_M) / CELLS_ACROSS, (m.height - 2 * FRAME_M) / CELLS_ALONG
    col, row = (ms - FRAME_M) // cw, (mt - FRAME_M) // ch
    if cell is not None:
        crop[(row == cell[0]) & (col == cell[1])] += dt
    if band is not None:
        crop[(col // 2) == band] += dt
    frame = (ms < FRAME_M) | (ms > m.width - FRAME_M) | (mt < FRAME_M) | (mt > m.height - FRAME_M)
    return np.where(frame, 34.0, crop)


@pytest.mark.parametrize("cell", [(0, 0), (5, 3), (11, 5), (7, 1)])
def test_hot_cell_is_located_to_its_cell(module, cell):
    a = analyse_crop(
        synthetic_crop(module, np.random.default_rng(1), cell=cell), module, Settings(), RES
    )
    assert a["substring"] == -1
    assert len(a["blobs"]) == 1 and a["blobs"][0]["cells"] == [cell]
    assert a["blobs"][0]["peak"] == pytest.approx(8.0, abs=0.3)


@pytest.mark.parametrize("band", [0, 1, 2])
def test_warm_third_is_a_bypass_diode_substring(module, band):
    a = analyse_crop(
        synthetic_crop(module, np.random.default_rng(2), band=band, dt=4.0), module, Settings(), RES
    )
    assert a["substring"] == band
    assert a["band_dt"] == pytest.approx(4.0, abs=0.2)
    assert a["blobs"] == []


def test_healthy_module_has_no_finding(module):
    a = analyse_crop(synthetic_crop(module, np.random.default_rng(3)), module, Settings(), RES)
    assert a["substring"] == -1 and a["blobs"] == []


def test_relative_mode_works_on_contrast_stretched_images(module):
    crop = synthetic_crop(module, np.random.default_rng(4), cell=(3, 2), dt=2.0, noise=0.3)
    stretched = np.round(255 * (crop - crop.min()) / (crop.max() - crop.min()))  # 8-bit, no kelvin
    a = analyse_crop(stretched, module, Settings(), RES, relative=True)
    assert [b["cells"] for b in a["blobs"]] == [[(3, 2)]]


def test_fusion_needs_most_views_to_agree(scene):
    mapper = Mapper(scene, Settings())
    mid = "R04-C050-1"
    hot = [View(mid, i, "cell", 9.0, [(2, 3)], 1) for i in range(2)]
    none = [View(mid, i + 2, "none") for i in range(3)]
    assert mapper.fuse(hot + none) == []  # 2 of 5 views
    f = mapper.fuse(hot + none[:2])  # 2 of 4 views
    assert [(x.module_id, x.kind, x.cells) for x in f] == [(mid, "hot_cell", [(2, 3)])]
    assert mapper.fuse([View(mid, 0, "cell", 9.0, [(2, 3)], 1)] + none[:1]) == []  # one view only


def test_neighbouring_offline_modules_are_one_string(scene):
    mapper = Mapper(scene, Settings())
    views = [View(f"R07-C{c:03d}-2", k, "offline", 4.0) for c in (20, 21, 22, 30) for k in range(3)]
    kinds = {f.module_id: f.kind for f in mapper.fuse(views)}
    assert kinds == {
        "R07-C020-2": "string",
        "R07-C021-2": "string",
        "R07-C022-2": "string",
        "R07-C030-2": "offline",
    }


def test_score_counts_next_door_false_positives():
    truth = [Defect("R01-C010-1", "hot_cell", 9.0, [(1, 1)]), Defect("R02-C005-2", "offline", 4.0)]
    from pvdefect.mapper import Finding

    found = [
        Finding("R01-C011-1", "hot_cell", 9.0, 3, 3, [(1, 1)]),
        Finding("R02-C005-2", "offline", 4.1, 3, 3),
    ]
    s = score(found, truth)
    assert (s.tp, s.fp, s.fn, s.next_door) == (1, 1, 1, 1)
    assert s.kind_accuracy == 1.0


def test_one_pass_finds_every_defect_on_its_row(scene, defects, rendered_pass):
    """End to end on one rendered pass, exact poses: right module, type and cell."""
    idx, imgs, true = rendered_pass
    findings, views = Mapper(scene, Settings()).run(
        [(img, true[i]) for img, i in zip(imgs, idx, strict=True)]
    )
    seen = {v.module_id for v in views}
    truth = [d for d in defects if d.module_id in seen]
    assert len(truth) >= 3
    s = score(findings, truth)
    assert (s.fp, s.fn) == (0, 0)
    assert s.kind_accuracy == 1.0
    found = {f.module_id: f for f in findings}
    for d in truth:
        if d.kind == "hot_cell":
            assert found[d.module_id].cells == d.cells
        assert found[d.module_id].delta_t == pytest.approx(d.delta_t, abs=1.5)
