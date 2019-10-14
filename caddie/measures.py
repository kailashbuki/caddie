#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module abstracts information measures."""
import abc
from collections import Counter
from enum import Enum
from math import log
from typing import List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import chi2_contingency    # type: ignore

from .scomp import stochastic_complexity

__all__ = ['DependenceMeasure',
           'DependenceMeasureType',
           'ChiSquaredTest',
           'ShannonEntropy',
           'StochasticComplexity']


class DependenceMeasureType(Enum):
    NHST: int = 1    # Null Hypothesis Significance Testing
    INFO: int = 2    # Information-theoretic


class DependenceMeasure(abc.ABC):

    @property
    @abc.abstractmethod
    def type(self) -> DependenceMeasureType:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def measure(seq1: List[int], seq2: Optional[List[int]] = None) -> float:
        raise NotImplementedError


class ShannonEntropy(DependenceMeasure):
    type: DependenceMeasureType = DependenceMeasureType.INFO

    @staticmethod
    def measure(seq1: List[int], seq2: Optional[List[int]] = None) -> float:
        entropy = 0.0
        n = len(seq1)
        counts = Counter(seq1).values()
        for count in counts:
            entropy -= (count / n) * (log(count, 2) - log(n, 2))
        return entropy


class StochasticComplexity(DependenceMeasure):
    type: DependenceMeasureType = DependenceMeasureType.INFO

    @staticmethod
    def measure(seq1: List[int], seq2: Optional[List[int]] = None) -> float:
        return stochastic_complexity(seq1)


class ChiSquaredTest(DependenceMeasure):
    type: DependenceMeasureType = DependenceMeasureType.NHST

    @staticmethod
    def measure(seq1: List[int], seq2: Optional[List[int]] = None) -> float:
        if not seq2:
            raise ValueError('seq2 is missing.')

        assert len(seq1) == len(seq2), "samples are not of the same size"

        dom_seq1 = list(set(seq1))
        dom_seq2 = list(set(seq2))

        ndom_seq1 = len(dom_seq1)
        ndom_seq2 = len(dom_seq2)

        indices1 = dict(zip(dom_seq1, range(ndom_seq1)))
        indices2 = dict(zip(dom_seq2, range(ndom_seq2)))

        table = np.zeros((ndom_seq1, ndom_seq2))

        for k, v1 in enumerate(seq1):
            v2 = seq2[k]
            i, j = indices1[v1], indices2[v2]
            table[i, j] += 1

        _, p_value, _, _ = chi2_contingency(table, correction=False)
        return p_value
