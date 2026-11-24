"""Page furniture drawn onto a page already dithered."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fugleramme.render import page
from fugleramme.render.page import Edges


@pytest.mark.parametrize("margin", [Edges.even(0), Edges(0.1, 0.05, 0.02, 0.2)])
def test_the_battery_mark_sits_in_the_top_right_inside_the_margin(margin):
    """Palette indices only, so no dither speckles it, and nothing past the margin."""
    plain = Image.new("P", (1200, 1600), 2)
    marked = page.battery(plain, margin)
    assert marked.mode == "P" and marked.size == plain.size
    assert set(np.unique(np.asarray(marked))) <= {0, 1, 2, 3}
    ys, xs = (np.asarray(marked) != 2).nonzero()
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    left, top, right, _bottom = margin.window(plain.size)
    assert top <= y0 and x1 < right
    assert x0 > (left + right) / 2 and y1 < 1600 / 4
