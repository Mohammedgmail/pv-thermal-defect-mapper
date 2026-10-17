import numpy as np
import pytest

from pvdefect.camera import rays
from pvdefect.refine import refine_poses
from pvdefect.simulate import GNSS, RTK, PoseErrors, logged_poses

from .conftest import LINE


def centre_error(scene, p, q):
    """Distance on the module plane between where two poses put the image centre."""
    row = scene.rows[LINE]

    def hit(pose):
        d = rays(np.array([319.5]), np.array([255.5]), pose, scene.camera)[0]
        return pose.position + ((row.center - pose.position) @ row.normal) / (d @ row.normal) * d

    return float(np.linalg.norm(hit(p) - hit(q)))


@pytest.mark.parametrize("errors,seed", [(RTK, 1), (GNSS, 1), (GNSS, 2)])
def test_refinement_puts_the_images_back_on_the_modules(scene, rendered_pass, errors, seed):
    idx, imgs, true = rendered_pass
    logged = logged_poses(true, errors, np.random.default_rng(10_000 + seed))
    refined, report = refine_poses(
        scene, [(img, logged[i]) for img, i in zip(imgs, idx, strict=True)]
    )
    before = np.array([centre_error(scene, logged[i], true[i]) for i in idx])
    after = np.array([centre_error(scene, q, true[i]) for q, i in zip(refined, idx, strict=True)])
    assert report[0].applied and report[0].row_end_frames >= 2
    assert np.median(after) < 0.05  # m, a quarter of a cell
    assert (after > 0.3).sum() <= 2
    assert np.median(after) < np.median(before)


def test_refinement_keeps_the_log_when_the_table_edge_is_out_of_reach(scene, rendered_pass):
    idx, imgs, true = rendered_pass
    far = PoseErrors(position_bias_m=0.0, position_jitter_m=0.0)
    logged = [p.moved((0.0, 0.0, 6.0)) for p in logged_poses(true, far, np.random.default_rng(0))]
    refined, report = refine_poses(
        scene, [(img, logged[i]) for img, i in zip(imgs, idx, strict=True)]
    )
    assert not report[0].applied
    assert all(
        np.allclose(q.position, logged[i].position) for q, i in zip(refined, idx, strict=True)
    )
