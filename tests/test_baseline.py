import numpy as np

from pvdefect.baseline import pnp_refine
from pvdefect.simulate import fixed_bias_poses

from .test_refine import centre_error


def test_pnp_fixes_small_errors_but_locks_onto_the_neighbour_module(scene, rendered_pass):
    idx, imgs, true = rendered_pass
    t = [true[i] for i in idx]
    small = fixed_bias_poses(t, np.random.default_rng(0), yaw_deg=0.3, pitch_deg=0.3)
    fixed = pnp_refine(scene, list(zip(imgs, small, strict=True)))
    before = np.median([centre_error(scene, p, q) for p, q in zip(small, t, strict=True)])
    after = np.median([centre_error(scene, p, q) for p, q in zip(fixed, t, strict=True)])
    assert after < 0.5 * before
    # 1.5 m off: every corner matches a neighbour, so the fit stays a module or more away
    big = fixed_bias_poses(t, np.random.default_rng(1), horizontal_m=1.5, vertical_m=1.5)
    fixed = pnp_refine(scene, list(zip(imgs, big, strict=True)))
    assert np.median([centre_error(scene, p, q) for p, q in zip(fixed, t, strict=True)]) > 0.5
