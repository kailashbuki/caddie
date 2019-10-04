#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the paper titled `MDL for Causal Inference on
Discrete Data`. For more detail, please refer to the manuscript at
http://people.mpi-inf.mpg.de/~kbudhath/manuscript/cisc.pdf
"""
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from .scomp import stochastic_complexity

__all__ = ['cisc', 'partition']


def partition(seq: List[int], by: List[int]) -> DefaultDict[int, List]:
    partitioned_seq: DefaultDict[int, List] = defaultdict(list)
    for i, by_obs in enumerate(by):
        partitioned_seq[by_obs].append(seq[i])
    return partitioned_seq


def cisc(X: List[int], Y: List[int]) -> Tuple[float, float]:
    assert len(X) == len(Y)

    n = len(X)

    scX = stochastic_complexity(X)
    scY = stochastic_complexity(Y)

    YgX = partition(Y, X)
    XgY = partition(X, Y)

    domX = YgX.keys()
    domY = XgY.keys()

    ndomX = len(domX)
    ndomY = len(domY)

    scYgX = sum(stochastic_complexity(Yp, ndomY) for Yp in YgX.values())
    scXgY = sum(stochastic_complexity(Xp, ndomX) for Xp in XgY.values())

    XtoY = scX + scYgX
    YtoX = scY + scXgY

    return (XtoY, YtoX)
