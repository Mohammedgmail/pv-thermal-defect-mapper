import numpy as np
import pytest
from scipy.integrate import quad

from pvdefect.errormodel import jacobian, offset_sd, p_right_cell, p_same_cell, plane_point, pose_sd
from pvdefect.simulate import GNSS, RTK


@pytest.mark.parametrize("sigma", [0.02, 0.1, 0.5, 2.0])
def test_closed_form_matches_numerical_integration(sigma):
    w = 0.18
    num = quad(
        lambda d: (
            max(0.0, 1 - abs(d) / w) * np.exp(-(d**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma
        ),
        -10 * sigma - w,
        10 * sigma + w,
        points=[-w, 0, w],
    )[0]
    assert p_same_cell(sigma, w) == pytest.approx(num, abs=1e-6)


def test_jacobian_follows_the_viewing_geometry(scene):
    pose, row = scene.poses[80], scene.rows[3]
    J = jacobian(pose, scene, row)
    centre = plane_point(pose, scene, row, 319.5, 255.5)
    horizontal = np.linalg.norm((centre - pose.position)[:2])
    # a heading error moves the image along the row by the horizontal range per radian
    assert abs(J[0, 3]) == pytest.approx(horizontal, rel=0.02)
    # a camera shift along the row moves the image by the same amount
    assert abs(J[0, 0]) == pytest.approx(1.0, abs=1e-3)
    # a pitch error moves it up the slope by more than the slant range per radian
    slant = np.linalg.norm(centre - pose.position)
    assert abs(J[1, 4]) > slant


def test_first_order_model_predicts_the_actual_offset(scene):
    pose, row = scene.poses[80], scene.rows[3]
    J = jacobian(pose, scene, row)
    err = np.array([0.05, -0.04, 0.03, np.radians(0.2), np.radians(-0.15)])
    moved = pose.moved(err[:3], np.degrees(err[3]), np.degrees(err[4]))
    d = plane_point(moved, scene, row, 319.5, 255.5) - plane_point(pose, scene, row, 319.5, 255.5)
    actual = np.array([d @ row.along, d @ row.upslope])
    assert actual == pytest.approx(J @ err, abs=0.01)


def test_rtk_attitude_is_too_coarse_for_cells_but_fine_for_modules(scene):
    pose, row = scene.poses[80], scene.rows[3]
    m = scene.module("R04-C010-1")
    sd_rtk = offset_sd(jacobian(pose, scene, row), pose_sd(RTK))
    assert p_right_cell(sd_rtk, m.cell_size) < 0.3
    assert p_right_cell(sd_rtk, (m.width, m.height)) > 0.75
    sd_gnss = offset_sd(jacobian(pose, scene, row), pose_sd(GNSS))
    assert p_right_cell(sd_gnss, (m.width, m.height)) < 0.3
