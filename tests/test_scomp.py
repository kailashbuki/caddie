#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from caddie import scomp


def test_parametric_complexity():
    np.testing.assert_almost_equal(scomp.parametric_complexity(16, 16), 18.11, decimal=1)
    assert scomp.parametric_complexity(1000, 1) == 0.0


def test_stochastic_complexity():
    assert scomp.stochastic_complexity([1]*1000) == 0.0
