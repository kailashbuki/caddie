#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module assess the performance of various discrete causal inference
methods on synthetic cause-effect pairs.
"""
from collections import defaultdict
import os
import sys
from typing import DefaultDict, Dict, List, Tuple, Type

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm   # type: ignore

from .anm import fit_anm_both_dir
from .cisc import cisc
from .measures import ChiSquaredTest, DependenceMeasure, ShannonEntropy, StochasticComplexity

__all__ = ['run_cisc', 'run_dr', 'run_it_anm',
           'simulate_accuracy_against_sample_size',
           'simulate_decision_rate_against_data_type']


def _dr_curve(decisions: np.array,
              score_diffs: np.array) -> Tuple[List[float], List[float]]:
    num_pairs = np.size(score_diffs)
    decision_rates, accuracies = [], []

    unq_score_diffs = np.unique(score_diffs)[::-1]
    for score_diff in unq_score_diffs:
        ok_pairs = score_diffs >= score_diff
        decision_ok_pairs = decisions[ok_pairs]

        num_ok_pairs = np.sum(ok_pairs)
        decision_rate = num_ok_pairs / num_pairs
        ncorrect_decisions = np.sum(decision_ok_pairs)
        accuracy = ncorrect_decisions / num_ok_pairs

        accuracies.append(accuracy)
        decision_rates.append(decision_rate)
    return decision_rates, accuracies


def _map_randomly(dom_f: List[int], img_f: List[int]) -> Dict[int, int]:
    f = dict((x, np.random.choice(img_f)) for x in dom_f)
    return f


def _gen_X(src: str, size: int) -> List[int]:
    if src == "uniform":
        max_X = np.random.randint(2, 10)
        X = np.array([np.random.randint(1, max_X) for i in range(size)])
    elif src == "multinomial":
        p_nums = [
            np.random.randint(1, 10) for i in range(np.random.randint(3, 11))
        ]
        p_vals = [v / sum(p_nums) for v in p_nums]
        X = np.random.multinomial(size, p_vals, 1)[0]
        X = [[i + 1] * f for i, f in enumerate(X)]
        X = np.array([j for sublist in X for j in sublist])
    elif src == "binomial":
        n = np.random.randint(1, 40)
        p = np.random.uniform(0.1, 0.9)
        X = np.random.binomial(n, p, size)
    elif src == "geometric":
        p = np.random.uniform(0.05, 0.95)
        X = np.random.geometric(p, size)
    elif src == "hypergeometric":
        ngood = np.random.randint(1, 40)
        nbad = np.random.randint(1, 40)
        nsample = np.random.randint(1, ngood + nbad)
        X = np.random.hypergeometric(ngood, nbad, nsample, size)
    elif src == "poisson":
        rate = np.random.randint(1, 10)
        X = np.random.poisson(rate, size)
    elif src == "negativeBinomial":
        n = np.random.randint(1, 40)
        p = np.random.uniform(0.1, 0.9)
        X = np.random.negative_binomial(n, p, size)
    X = X.astype(int)
    return X


def _gen_N(size: int) -> List[int]:
    t = np.random.randint(1, 8)
    N = np.array([np.random.randint(-t, t + 1) for i in range(size)])
    N = N.astype(int)
    return N


def _gen_XY(srcX: str,
            supp_fX: List[int],
            size: int) -> Tuple[List[int], List[int]]:
    X = _gen_X(srcX, size)
    supp_X = list(set(X))
    f = _map_randomly(supp_X, supp_fX)
    N = _gen_N(size)
    Y = [f[X[i]] + N[i] for i in range(size)]
    return X, Y


def _validate_dir(dirpath: str) -> None:
    if not os.path.isdir(dirpath):
        raise IOError('Invalid directory: %s' % dirpath)


def run_cisc(X: List[int], Y: List[int]) -> Tuple[bool, bool, float]:
    cisc_score = cisc(X, Y)
    decision = cisc_score[0] != cisc_score[1]
    XtoY = cisc_score[0] < cisc_score[1]
    diff = abs(cisc_score[0] - cisc_score[1])
    return decision, XtoY, diff


def run_dr(X: List[int],
           Y: List[int],
           level: float,
           max_niter: int) -> Tuple[bool, bool, float]:
    dr_score = fit_anm_both_dir(
        X, Y, ChiSquaredTest, max_niter=max_niter, level=level)
    indep_XtoY_only = dr_score[0] > level and dr_score[1] < level
    indep_YtoX_only = dr_score[0] < level and dr_score[1] > level
    decision = indep_XtoY_only or indep_YtoX_only
    XtoY = indep_XtoY_only
    diff = abs(dr_score[0] - dr_score[1])
    return decision, XtoY, diff


def run_it_anm(X: List[int],
               Y: List[int],
               measure: Type[DependenceMeasure],
               max_niter: int) -> Tuple[bool, bool, float]:
    score = fit_anm_both_dir(X, Y, measure, max_niter=max_niter)
    decision = score[0] != score[1]
    XtoY = score[0] < score[1]
    diff = abs(score[0] - score[1])
    return decision, XtoY, diff


def simulate_accuracy_against_sample_size(results_dir: str,
                                          nsample: int = 500,
                                          srcX: str = 'geometric',
                                          max_niter: int = 10,
                                          dr_level: float = 0.05) -> None:
    _validate_dir(results_dir)

    supp_fX = list(range(-7, 8))
    sample_sizes = [100, 200, 400, 800, 1600]
    results = pd.DataFrame(0.0,
                           index=np.arange(len(sample_sizes)),
                           columns=['sample_size', 'acc_cisc', 'acc_dr',
                                    'acc_acid', 'acc_crisp', 'dec_cisc',
                                    'dec_dr', 'dec_acid', 'dec_crisp'])

    for i, sample_size in enumerate(tqdm(sample_sizes)):
        decisions_by_method: DefaultDict[str, List[bool]] = defaultdict(list)

        for j in range(nsample):
            X, Y = _gen_XY(srcX, supp_fX, sample_size)

            cisc_dec, cisc_XtoY, _ = run_cisc(X, Y)
            dr_dec, dr_XtoY, _ = run_dr(X, Y, dr_level, max_niter)
            acid_dec, acid_XtoY, _ = run_it_anm(
                X, Y, ShannonEntropy, max_niter)
            crisp_dec, crisp_XtoY, _ = run_it_anm(
                X, Y, StochasticComplexity, max_niter)

            if cisc_dec:
                decisions_by_method['cisc'].append(cisc_XtoY)
            if dr_dec:
                decisions_by_method['dr'].append(dr_XtoY)
            if acid_dec:
                decisions_by_method['acid'].append(acid_XtoY)
            if crisp_dec:
                decisions_by_method['crisp'].append(crisp_XtoY)

        results.loc[i]['sample_size'] = sample_size
        for method in decisions_by_method:
            decs = decisions_by_method[method]
            results.loc[i]['acc_%s' % method] = sum(decs) / len(decs)
            results.loc[i]['dec_%s' % method] = len(decs) / nsample
    print(results)
    results.to_csv(os.path.join(
        results_dir, 'acc_dec_by_sample_size.csv'), index=False)


def simulate_decision_rate_against_data_type(results_dir: str,
                                             nsample: int = 500,
                                             sample_size=1000,
                                             max_niter: int = 10,
                                             dr_level: float = 0.05) -> None:
    _validate_dir(results_dir)

    supp_fX = list(range(-7, 8))
    srcsX = ['uniform', 'binomial', 'negativeBinomial', 'geometric',
             'hypergeometric', 'poisson', 'multinomial']
    for srcX in tqdm(srcsX):
        decs_by_method: DefaultDict[str, List[bool]] = defaultdict(list)
        diffs_by_method: DefaultDict[str, List[float]] = defaultdict(list)

        for i in range(nsample):
            X, Y = _gen_XY(srcX, supp_fX, sample_size)

            cisc_dec, cisc_XtoY, cisc_diff = run_cisc(X, Y)
            dr_dec, dr_XtoY, dr_diff = run_dr(X, Y, dr_level, max_niter)
            acid_dec, acid_XtoY, acid_diff = run_it_anm(
                X, Y, ShannonEntropy, max_niter)
            crisp_dec, crisp_XtoY, crisp_diff = run_it_anm(
                X, Y, StochasticComplexity, max_niter)

            if cisc_dec:
                decs_by_method['cisc'].append(cisc_XtoY)
                diffs_by_method['cisc'].append(cisc_diff)
            if dr_dec:
                decs_by_method['dr'].append(dr_XtoY)
                diffs_by_method['dr'].append(dr_diff)
            if acid_dec:
                decs_by_method['acid'].append(acid_XtoY)
                diffs_by_method['acid'].append(acid_diff)
            if crisp_dec:
                decs_by_method['crisp'].append(crisp_XtoY)
                diffs_by_method['crisp'].append(crisp_diff)

        for method in decs_by_method:
            rates, accs = _dr_curve(
                np.array(decs_by_method[method]), np.array(diffs_by_method[method]))
            df = pd.DataFrame(data=dict(rates=rates, accuracies=accs))
            df.to_csv(os.path.join(results_dir, '%s-drate-%s.csv' %
                                   (srcX, method)), sep=',', index=False)
