# bnsobol - Variance-based Sensitivity Analysis for Bayesian Networks

This library computes the main Sobol indices (that is, the *variance components* and the *total indices* [1]) of a function $f$ that is encoded by a Bayesian network. The functions supported are such that:

- Their inputs are a subset of nodes of the network;
- Their output is the expected value of one of the networks' nodes.

References:

[1] Saltelli, A. et al.: "Global Sensitivity Analysis: The Primer" (2008)

## Why *bnsobol*?

Estimating Sobol indices is computationally hard, with brute-force or Monte Carlo estimation methods usually requiring millions of samples. In our case, each sample would require inference in a Bayesian network, which has a significant computational cost. Instead of expensive sampling, this method exploits the network structure and can compute each Sobol index exactly using a few network marginalization queries only. Running times for *bnsobol* will of course depend on each network's topology, but you can expect <1 second per index for most networks with <100 variables.

## Installation

You can install *bnsobol* from the source as follows:

```
git clone https://github.com/rballester/bnsobol.git
cd bnsobol
pip install .
```

**Dependences**: [*NumPy*](https://numpy.org/), [*pgmpy*](https://github.com/pgmpy/pgmpy)

## Example

To see a usage example, see this [Jupyter notebook](https://github.com/rballester/bnsobol/blob/master/examples/concrete.ipynb).

## Tests

We use [*pytest*](https://docs.pytest.org/en/latest/), and the tests depend on [*tntorch*](https://github.com/rballester/tntorch). To run them, do:

```
cd tests/
pytest
```

## Citing

If you use *bnsobol* for your research, please cite the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0951832022002204):

```
@article{BL:22,
  title={Computing Sobol indices in probabilistic graphical models},
  author={Ballester-Ripoll, Rafael and Leonelli, Manuele},
  journal={Reliability Engineering and System Safety},
  volume={to appear},
  year={2022}
}
```

## Contributing

Pull requests are welcome!

Besides using the [issue tracker](https://github.com/rballester/bnsobol/issues), feel also free to contact me at <rafael.ballester@ie.edu>.
