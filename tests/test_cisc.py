#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

from caddie import cisc


def test_cisc():
    X = [random.randint(-5, 6) for i in range(2000)]
    Y = X

    XtoY, YtoX = cisc.cisc(X, Y)
    assert XtoY == YtoX
