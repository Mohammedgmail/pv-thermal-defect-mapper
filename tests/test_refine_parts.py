"""The building blocks of the pose refinement, on synthetic data with a known answer."""

import numpy as np
from pvmission.plant import Row

from pvdefect import refine as RF
from pvdefect.camera import project
from pvdefect.mapper import _hidden

ROW = Row(0, np.array([0.0, 0.0, 1.5]), 30 * 1.0, 3.5, 20.0, 180.0, 30, 2)  # 1 m modules
W = ROW.length / ROW.modules_along


def test_nanmedian_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 60))
    x[rng.random(x.shape) < 0.3] = np.nan
    x[5] = np.nan  # an all-NaN row
    for axis in (0, 1):
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.nanmedian(x, axis=axis)
        np.testing.assert_array_equal(RF._nanmedian(x, axis), ref)
    assert RF._nanmedian(x, None) == np.nanmedian(x)


def test_fast_ortho_matches_the_direct_computation(scene, rendered_pass):
    """The plane-affine projection and the separable hidden mask give the same ortho as
    projecting every 3-D point and testing every line of sight."""
    idx, imgs, true = rendered_pass
    pose, img = true[idx[3]], imgs[3]
    row = scene.rows[3]
    s = np.arange(-12, 12, RF.RES) + (pose.position - row.center) @ row.along
    t = np.arange(-row.width / 2 - 2, row.width / 2 + 2, RF.RES)
    fast = RF.ortho(img, pose, scene, row, s, t)
    S, T = np.meshgrid(s, t)
    P = row.point(S, T)
    u, v, depth = project(P, pose, scene.camera)
    cam = scene.camera
    from scipy import ndimage

    ref = ndimage.map_coordinates(img, [v, u], order=1, mode="constant", cval=np.nan)
    bad = (depth <= 0) | (u < 0) | (v < 0) | (u > cam.width_px - 1) | (v > cam.height_px - 1)
    near = [r for r in scene.rows if r is not row and abs(r.index - row.index) <= 2]
    bad |= _hidden(P, pose.position, near)
    ref = np.where(bad, np.nan, ref)
    assert (np.isnan(fast) == np.isnan(ref)).mean() > 0.999  # boundary pixels may differ
    both = np.isfinite(fast) & np.isfinite(ref)
    assert np.abs(fast[both] - ref[both]).max() < 1e-4


def _profile_with_end(s, end_s, ground_k, line_k=-3.0, module_k=40.0):
    """Along-row temperature: modules with cool frame lines up to ``end_s``, ground beyond."""
    x = (s - (-ROW.length / 2)) % W
    on_line = ((x < 0.035) | (x > W - 0.035)) & (s < end_s - 0.1)  # outer frame blends in
    prof = np.where(on_line, module_k + line_k, module_k)
    return np.where(s < end_s, prof, ground_k)


def test_row_end_found_from_the_frame_lines_when_the_ground_is_as_warm():
    rng = np.random.default_rng(1)
    s = np.arange(10.0, 22.0, RF.RES)  # the frame sees the +L/2 end (15 m) at 1.3 m offset
    true_end = ROW.length / 2 + 1.3
    for ground_k, temp_end in ((25.0, True), (39.5, False)):
        prof = _profile_with_end(s - 1.3, true_end - 1.3, ground_k) + rng.normal(0, 0.1, len(s))
        o = np.tile(prof, (20, 1))
        found_t = RF._row_end(o, s, ROW, 4.0)
        assert (found_t is not None) == temp_end
        e_mod = 0.3  # 1.3 m = 0.3 + 1 module
        found_p = RF._row_end_pattern(o, s, ROW, e_mod, 4.0)
        assert found_p is not None and abs(found_p - 1.3) < 0.05


def test_lower_edge_found_in_raw_counts_with_a_soft_edge():
    t = np.arange(-ROW.width / 2 - 3, ROW.width / 2 + 3, RF.RES)
    true = 0.4  # table shifted 40 cm up the slope
    edge = -ROW.width / 2 + true
    # 4 K step spread over 60 cm (dirt and frame at the lower edge, lens blur)
    prof = 280 + 4.0 * np.clip((t - edge + 0.3) / 0.6, 0, 1)
    prof = np.where(t > ROW.width / 2 + true, 277.0, prof)  # shadow behind the row
    o = np.tile(prof[:, None], (1, 50))
    got = RF._lower_edge(o, t, ROW, 2.5)
    assert got is not None and abs(got - true) < 0.2


def test_texture_links_carry_the_row_end_through_a_gnss_jump():
    """Ten frames; the along-row error jumps by 0.9 m after frame 3 (quantised GNSS), only
    the last frame sees the row end, and one frame has a wrong anchor. Interpolating the
    anchors gets frames 0-3 wrong; the texture links between frames get them all right."""
    rng = np.random.default_rng(2)
    world = rng.normal(0, 0.5, (40, 4000))  # module texture on a 3 cm grid, 120 m of row
    world_s = np.arange(4000) * RF.RES - 60
    t_tab = np.linspace(-1.6, 1.6, 40)
    true_e = np.array([0.95] * 4 + [0.05] * 6)
    ks = list(range(10))
    hp, s_of, emod = {}, {}, {}
    for k in ks:
        cam_s = -8 + 1.5 * k
        s = np.arange(cam_s - 6, cam_s + 6, RF.RES)
        j = np.round((s - true_e[k] - world_s[0]) / RF.RES).astype(int)  # seen at s + e
        hp[k] = world[:, j] + rng.normal(0, 0.05, (40, len(s)))
        s_of[k] = s
        emod[k] = (true_e[k] + W / 2) % W - W / 2
    ends = {k: None for k in ks}
    ends[8] = ends[9] = 0.05
    ends[6] = 1.05  # wrong by a module
    without = RF._module_steps(ks, emod, ends, hp, s_of, t_tab, W, 4.0, ROW, chain=False)
    with_ = RF._module_steps(ks, emod, ends, hp, s_of, t_tab, W, 4.0, ROW, chain=True)
    e_without = np.array([emod[k] + without[k] * W for k in ks])
    e_with = np.array([emod[k] + with_[k] * W for k in ks])
    assert np.abs(e_without - true_e).max() > 0.5
    np.testing.assert_allclose(e_with, true_e, atol=1e-9)
