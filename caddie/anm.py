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
                        max_niter: int,
                        level: float) -> float:
    f = _map_x_to_freq_y(X, Y)
    pair = list(zip(X, Y))
    eps = [y - f[x] for x, y in pair]
    cur_eps_info = dep_measure.measure(eps, X)

    supp_X = list(set(X))
    supp_img_f = list(range(min(Y), max(Y)+1))
    # maximise the p-value, but minimise other information-theoretic scores
    opt_prob = min if dep_measure.type == DependenceMeasureType.INFO else max

    if not (len(supp_X) == 1 or len(supp_img_f) == 1):
        j = 0
        while j < max_niter:
            # if the dep_measure is a NHST, then cur_eps_info is a p-value.
            if dep_measure.type == DependenceMeasureType.NHST and cur_eps_info > level:
                break

            # optimise in one direction at a time
            random.shuffle(supp_X)
            for x_to_map in supp_X:
                best_img_fx = None

                for cand_img_fx in supp_img_f:
                    if cand_img_fx == f[x_to_map]:
                        continue

                    eps = [y - f[x] if x != x_to_map else y -
                           cand_img_fx for x, y in pair]
                    new_eps_info = dep_measure.measure(eps, X)
                    if opt_prob(new_eps_info, cur_eps_info) == new_eps_info:
                        cur_eps_info = new_eps_info
                        best_img_fx = cand_img_fx

                # update f if the dep measure is optimised in this direction
                if best_img_fx:
                    f[x_to_map] = best_img_fx
            j += 1

    if dep_measure.type == DependenceMeasureType.INFO:
        return dep_measure.measure(X) + cur_eps_info
    else:
        return cur_eps_info


def fit_anm_both_dir(X: List[int], Y: List[int],
                     dep_measure: Type[DependenceMeasure],
                     max_niter: int = 10,
                     level: float = 0.05) -> Tuple[float, float]:
    assert issubclass(dep_measure, DependenceMeasure), "dependence measure " \
                                                       "must be a subclass of DependenceMeasure abstract class"
    XtoY = discrete_regression(X, Y, dep_measure, max_niter, level)
    YtoX = discrete_regression(Y, X, dep_measure, max_niter, level)
    return (XtoY, YtoX)
