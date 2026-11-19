"""EXP-011: wider cam0 pinhole rectification (dvslam/data/tumvi.py, wide_*): same rays in the central crop."""
import numpy as np
import pytest

from dvslam.data import tumvi

INTR = np.array([190.97, 190.97, 254.93, 256.89])
DIST = np.array([0.0035, 0.0007, -0.0021, 0.0002])


@pytest.mark.parametrize('size', [640, 768, 1024])
def test_central_crop_of_wide_maps_equals_standard_maps(size):
    o = tumvi.wide_offset(size)
    mx, my = tumvi.rectify_maps(INTR, DIST)
    wx, wy = tumvi.rectify_maps_wide(INTR, DIST, size)
    assert wx.shape == (size, size)
    np.testing.assert_allclose(wx[o:o + 512, o:o + 512], mx, atol=1e-3)
    np.testing.assert_allclose(wy[o:o + 512, o:o + 512], my, atol=1e-3)


def test_wide_K_and_paths():
    K = tumvi.wide_K(768)
    assert K[0, 0] == 190.0 and K[0, 2] == 255.5 + 128 and K[1, 2] == 255.5 + 128
    with pytest.raises(ValueError):
        tumvi.wide_offset(513)
    p = tumvi.wide_path('/c/tumvi/room5/cam0_pinhole_512_f190/123.png', 768)
    assert str(p) == '/c/tumvi/room5/cam0_pinhole_768_f190/123.png'
