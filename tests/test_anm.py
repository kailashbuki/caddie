#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

from caddie import anm, measures


def test_fit_anm_both_dir():
    X = [random.randint(-5, 6) for i in range(2000)]
    Y = X

    XtoY, YtoX = anm.fit_anm_both_dir(X, Y, measures.ChiSquaredTest)
    assert XtoY == YtoX

    XtoY, YtoX = anm.fit_anm_both_dir(X, Y, measures.ShannonEntropy)
    assert XtoY == YtoX

    XtoY, YtoX = anm.fit_anm_both_dir(X, Y, measures.StochasticComplexity)
    assert XtoY == YtoX

    Y = [x + random.randint(-2, 3) for x in X]
    XtoY, YtoX = anm.fit_anm_both_dir(X, Y, measures.ShannonEntropy)
    assert XtoY < YtoX
