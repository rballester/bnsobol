# bnsobol - Variance-based Sensitivity Analysis for Bayesian Networks

This library computes the main Sobol indices (that is, the *variance components* and the *total indices* [1]) of a function $f$ that is encoded by a Bayesian network.

- Its inputs are a subset of nodes of the network;
- Its output is the expected value of one of the networks' nodes.

References:

[1] Saltelli, A. et al.: "Global Sensitivity Analysis: The Primer" (2008)

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

## Contributing

Pull requests are welcome!

Besides using the [issue tracker](https://github.com/rballester/bnsobol/issues), feel also free to contact me at <rafael.ballester@ie.edu>.
