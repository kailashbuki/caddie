#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements discrete regression.
"""
from collections import defaultdict, Counter
import random
import sys
from typing import Dict, List, Optional, Tuple, Type

from .cisc import partition
from .measures import DependenceMeasure, DependenceMeasureType

__all__ = ['discrete_regression', 'fit_anm_both_dir']


def _map_x_to_freq_y(X: List[int], Y: List[int]) -> Dict[int, int]:
    y_by_x = dict()
    Yprime_by_x = partition(Y, X)
    for x, Yprime in Yprime_by_x.items():
        frequent_y, _ = Counter(Yprime).most_common(1)[0]
        y_by_x[x] = frequent_y
    return y_by_x


def discrete_regression(X: List[int], Y: List[int],
                        dep_measure: Type[DependenceMeasure],
                        max_niter: int, level: float) -> float:
    supp_X = list(set(X))
    supp_Y = list(range(min(Y), max(Y)+1))
    f = _map_x_to_freq_y(X, Y)

    pair = list(zip(X, Y))
    res = [y - f[x] for x, y in pair]
    cur_res_inf = dep_measure.measure(res, X)
    sign = -1 if dep_measure.type == DependenceMeasureType.NHST else 1
    cur_res_inf *= sign

    j = 0
    while j < max_niter:
        if dep_measure.type == DependenceMeasureType.NHST and -cur_res_inf > level:
            break

        random.shuffle(supp_X)
        for x_to_map in supp_X:
            best_res_inf = sys.float_info.max

            for cand_y in supp_Y:
                if cand_y == f[x_to_map]:
                    continue

                res = [y - f[x] if x != x_to_map else y -
                       cand_y for x, y in pair]
                res_inf = dep_measure.measure(res, X)
                res_inf *= sign

                if res_inf < best_res_inf:
                    best_res_inf = res_inf
                    best_y = cand_y

            if best_res_inf < cur_res_inf:
                cur_res_inf = best_res_inf
                f[x_to_map] = best_y
        j += 1

    if dep_measure.type == DependenceMeasureType.INFO:
        return dep_measure.measure(X) + cur_res_inf
    else:
        return sign * cur_res_inf


def fit_anm_both_dir(X: List[int], Y: List[int],
                     dep_measure: Type[DependenceMeasure],
                     max_niter: int = 10,
                     level: float = 0.05) -> Tuple[float, float]:
    assert issubclass(dep_measure, DependenceMeasure), "dependence measure " \
                                                       "must be a subclass of DependenceMeasure abstract class"
    XtoY = discrete_regression(X, Y, dep_measure, max_niter, level)
    YtoX = discrete_regression(Y, X, dep_measure, max_niter, level)
    return (XtoY, YtoX)
