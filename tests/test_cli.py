import json

from pvdefect.cli import main

from .conftest import NOON


def test_simulate_then_map_with_refinement(tmp_path, capsys):
    survey, out = tmp_path / "survey", tmp_path / "results"
    main(["simulate", str(NOON), "--out", str(survey), "--lines", "3", "--errors", "gnss"])
    assert len(list((survey / "frames").glob("*.tiff"))) == 25
    main(["map", str(survey), "--mission", str(NOON), "--out", str(out), "--refine"])
    text = capsys.readouterr().out
    assert "Pose refinement: 1 of 1 passes corrected" in text
    for name in (
        "findings.csv",
        "findings.geojson",
        "report.md",
        "gallery.png",
        "refine.json",
        "score.json",
    ):
        assert (out / name).exists(), name
    score = json.loads((out / "score.json").read_text())
    assert score["f1"] == 1.0
    gj = json.loads((out / "findings.geojson").read_text())
    assert len(gj["features"]) == score["tp"]
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert 54 < ring[0][1] < 56  # Northumberland
