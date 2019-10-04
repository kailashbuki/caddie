#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the linear algorithm for computing the stochastic
complexity of a discrete sequence relative to a parametric family of
multinomial distributions. For more detail, please refer to
http://pgm08.cs.aau.dk/Papers/31_Paper.pdf
"""
from collections import Counter
from math import ceil, sqrt
from typing import Dict, List, Optional

import numpy as np


def parametric_complexity(n: int, k: int) -> float:
    total = 1.0
    b = 1.0
    d = 10

    bound = int(ceil(2 + sqrt(2 * n * d * np.log(10))))  # using equation (38)
    for i in range(1, bound + 1):
        b = (n - i + 1) / n * b
        total += b

    log_old_sum = np.log2(1.0)
    log_total = np.log2(total)
    log_n = np.log2(n)
    for i in range(3, k + 1):
        log_x = log_n + log_old_sum - log_total - np.log2(i - 2)
        x = 2 ** log_x
        log_one_plus_x = np.log2(1 + x)
        log_new_sum = log_total + log_one_plus_x
        log_old_sum = log_total
        log_total = log_new_sum

    if k == 1:
        log_total = np.log2(1.0)

    return log_total


def stochastic_complexity(X: List[int], k: Optional[int] = None) -> float:
    counts = Counter(X).values()
    n = len(X)
    k = k or len(counts)
    ml_code = 0.0
    for count in counts:
        ml_code += count * (np.log2(n) - np.log2(count))
    return ml_code + parametric_complexity(n, k)
