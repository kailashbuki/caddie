Caddie
-------

Caddie is a collection of bivariate discrete causal inference methods based on information-theoretic Additive Noise Models (ANM) and MDL-based instantiation of Algorithmic Independence of Conditionals (AIC).

Caddie Module Installation
----------------------------

The recommended way to install the `caddie` module is to simply use `pip`:

```console
$ pip install caddie
```
Caddie officially supports Python >= 3.6.

How to use caddie?
------------------
```pycon
>>> X = [1] * 1000
>>> Y = [-1] * 1000
>>> from caddie import cisc
>>> cisc.cisc(X, Y)                                                   # CISC
(0.0, 0.0)
>>> from caddie import anm, measures
>>> anm.fit_both_dir(X, Y, measures.StochasticComplexity)             # CRISP
(0.0, 0.0)
>>> anm.fit_both_dir(X, Y, measures.ChiSquaredTest)                   # DR
(1.0, 1.0)
>>> anm.fit_both_dir(X, Y, measures.ShannonEntropy)                   # ACID
(0.0, 0.0)
>>> from caddie import simulations
>>> simulations.simulate_decision_rate_against_data_type('/results/dir/') # for decision rate vs data type plots
...
>>> simulations.simulate_accuracy_against_sample_size('/results/dir/')    # for accuracy/decidability vs sample size plots
...
```

How to cite the paper?
----------------------
Todo: Add the citation to thesis.
