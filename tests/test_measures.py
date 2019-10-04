#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import pytest

from caddie import measures


def test_abstract_class():
    with pytest.raises(TypeError):
        measures.DependenceMeasure()

    with pytest.raises(NotImplementedError):
        measures.DependenceMeasure.measure([])


def test_chi_squared_test():
    # fully independent sample
    X = [1, 2, 1, 2]
    Y = [-1, -2, -2, -1]
    assert measures.ChiSquaredTest.type == measures.DependenceMeasureType.NHST
    assert measures.ChiSquaredTest.measure(X, Y) == 1.0

    with pytest.raises(ValueError):
        measures.ChiSquaredTest.measure(X)


def test_shannon_entropy():
    X = [1, 1, 1, 1, 2, 2, 2, 2]
    assert measures.ShannonEntropy.type == measures.DependenceMeasureType.INFO
    assert measures.ShannonEntropy.measure(X) == 1.0


def test_stochastic_complexity():
    X = [1, 1, 1, 1, 2, 2, 2, 2]
    assert measures.StochasticComplexity.type == measures.DependenceMeasureType.INFO
    print(measures.StochasticComplexity.measure(X))
