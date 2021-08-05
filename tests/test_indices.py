"""
Note: these tests require `tntorch` (https://github.com/rballester/tntorch) to compute the groundtruth Sobol indices
"""

import bnsobol as bn
import numpy as np
import torch
import tntorch as tn
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


def five_nodes():
    """
    Toy BN with two correlated inputs.

    :return: a `BayesianModel`
    """

    g = BayesianModel([
        ('A', 'B'),
        ('A', 'C'),
        ('C', 'B'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'D'),
        ('D', 'E'),
    ])

    sh = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7}

    tables = [
        ['A'],
        ['A', 'C'],
        ['A', 'C', 'B'],
        ['B', 'C', 'D'],
        ['B', 'D', 'E']
    ]
    for table in tables:
        t = np.random.rand(*[sh[var] for var in table])
        t /= np.sum(t, axis=-1, keepdims=True)
        cpd = TabularCPD(variable=table[-1], variable_card=sh[table[-1]],
                            values=t.reshape([-1, sh[table[-1]]]).T,
                            evidence=table[:-1],
                            evidence_card=[sh[var] for var in table[:-1]])
        g.add_cpds(cpd)

    g.check_model()

    return g


def five_nodes_uncorrelated():
    """
    Toy BN with two uncorrelated inputs.

    :return: a `BayesianModel`
    """

    g = BayesianModel([
        # ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'D'),
        ('D', 'E'),
    ])

    sh = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7}

    tables = [
        ['A'],
        ['B'],
        ['A', 'C'],
        ['B', 'C', 'D'],
        ['B', 'D', 'E']
    ]
    for table in tables:
        t = np.random.rand(*[sh[var] for var in table])
        t /= np.sum(t, axis=-1, keepdims=True)
        cpd = TabularCPD(variable=table[-1], variable_card=sh[table[-1]],
                            values=t.reshape([-1, sh[table[-1]]]).T,
                            evidence=table[:-1],
                            evidence_card=[sh[var] for var in table[:-1]])
        g.add_cpds(cpd)

    g.check_model()

    return g


def five_nodes_uniform():
    """
    Toy BN with two correlated inputs.

    :return: a `BayesianModel`
    """

    g = BayesianModel([
        ('A', 'C'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'D'),
        ('D', 'E'),
    ])

    sh = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7}

    tables = [
        ['A'],
        ['B'],
        ['A', 'C'],
        ['B', 'C', 'D'],
        ['B', 'D', 'E']
    ]
    for table in tables:
        if len(table) > 1:
            t = np.random.rand(*[sh[var] for var in table])
        else:
            t = np.ones([sh[table[0]]])
        t /= np.sum(t, axis=-1, keepdims=True)
        cpd = TabularCPD(variable=table[-1], variable_card=sh[table[-1]],
                            values=t.reshape([-1, sh[table[-1]]]).T,
                            evidence=table[:-1],
                            evidence_card=[sh[var] for var in table[:-1]])
        g.add_cpds(cpd)

    g.check_model()

    return g


b = five_nodes_uncorrelated()
# b = five_nodes_uniform()
# b = five_nodes()  # This one is not good for testing, since tntorch does not support Sobol for correlated inputs
inputs = ['A', 'B']
output = 'E'
values = np.arange(b.get_cardinality(output))
m = bn.util.to_mrf(b, output, values)


def equal(a, b):
    return np.abs(a-b)/np.abs(a) < 1e-2


def test_variance():

    p = bn.util.eliminate(b, to_keep=inputs, output='factor').values
    f = bn.util.eliminate(m, to_keep=inputs, output='factor').values / p
    gt = np.sum(p * f ** 2) - np.sum(p * f) ** 2
    V = bn.indices.variance(m, b, inputs)
    print(gt, V)
    assert equal(gt, V)


def test_variance_component():

    Si = bn.indices.variance_component(m, b, inputs, 'A')

    pab = bn.util.eliminate(b, to_keep=inputs, output='factor').values
    f = bn.util.eliminate(m, to_keep=inputs, output='factor').values / pab
    marg_a = np.sum(pab, axis=1)
    marg_b = np.sum(pab, axis=0)
    s = tn.symbols(len(inputs))
    gt = tn.sobol(tn.Tensor(f), tn.only(s[0]), marginals=[torch.Tensor(marg_a), torch.Tensor(marg_b)]).item()
    assert equal(gt, Si)


def test_higher_order():

    Sa = bn.indices.variance_component(m, b, inputs, ['A'])
    Sb = bn.indices.variance_component(m, b, inputs, ['B'])
    Sab = bn.indices.variance_component(m, b, inputs, ['A', 'B']) - Sa - Sb

    pab = bn.util.eliminate(b, to_keep=inputs, output='factor').values
    f = bn.util.eliminate(m, to_keep=inputs, output='factor').values / pab
    marg_a = np.sum(pab, axis=1)
    marg_b = np.sum(pab, axis=0)
    s = tn.symbols(len(inputs))
    gt = tn.sobol(tn.Tensor(f), tn.only(s[0]&s[1]), marginals=[torch.Tensor(marg_a), torch.Tensor(marg_b)]).item()
    assert equal(gt, Sab)


def test_total_index():

    STi = bn.indices.total_index(m, b, inputs, 'A')

    pab = bn.util.eliminate(b, to_keep=inputs, output='factor').values
    f = bn.util.eliminate(m, to_keep=inputs, output='factor').values / pab
    marg_a = np.sum(pab, axis=1)
    marg_b = np.sum(pab, axis=0)
    s = tn.symbols(len(inputs))
    gt = tn.sobol(tn.Tensor(f), s[0], marginals=[torch.Tensor(marg_a), torch.Tensor(marg_b)]).item()
    assert equal(gt, STi)


# def test_elimination():
#     print(m.get_partition_function())
#     factor = bn.indices.query(m, [], heuristic='MinNeighbors')
#     print(factor)
