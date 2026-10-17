import json

import numpy as np
from PIL import Image

from pvdefect.mapper import Settings
from pvdefect.realdata import classify, evaluate


def crop_24x40(rng, kind):
    """A Raptor-style crop: 24 x 40 px module with a dark border, 8-bit, stretched."""
    img = 150 + rng.normal(0, 3, (40, 24))
    if kind == "hot":
        img[10:13, 8:11] += 60
    elif kind == "diode":
        img[:, 16:24] += 25
    img[:2], img[-2:], img[:, :2], img[:, -2:] = 90, 90, 90, 90
    return np.clip(img, 0, 255)


def test_classify_tells_hot_spots_diodes_and_healthy_modules_apart():
    rng = np.random.default_rng(0)
    assert classify(crop_24x40(rng, "hot"), Settings()) == "cell"
    assert classify(crop_24x40(rng, "diode"), Settings()) == "substring"
    assert classify(crop_24x40(rng, "none"), Settings(hot_k=4)) == "none"


def test_evaluate_reads_the_dataset_layout(tmp_path):
    rng = np.random.default_rng(1)
    (tmp_path / "images").mkdir()
    meta = {}
    for i, (cls, kind) in enumerate(
        [("Cell", "hot")] * 4 + [("Diode", "diode")] * 4 + [("No-Anomaly", "none")] * 4
    ):
        Image.fromarray(crop_24x40(rng, kind).astype(np.uint8)).save(
            tmp_path / "images" / f"{i}.jpg", quality=95
        )
        meta[str(i)] = {"image_filepath": f"images/{i}.jpg", "anomaly_class": cls}
    (tmp_path / "module_metadata.json").write_text(json.dumps(meta))
    res = evaluate(tmp_path, Settings(hot_k=4), hot_k=(4,), band_k=(2.0,))
    assert res["images"] == 12
    assert res["default"]["Cell"]["cell"] == 1.0
    assert res["default"]["Diode"]["substring"] == 1.0
    assert res["default"]["No-Anomaly"].get("none") == 1.0
    assert res["sweep_hot"][0]["false_alarm"] == 0.0
