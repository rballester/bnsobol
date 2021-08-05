import numpy as np
import copy
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
import bnsobol as bn


def variance(m, b, inputs):
    """
    Compute the variance of an MRF `m` with respect to probability
    `b` and variables in `inputs`

    :param m: a `MarkovModel` encoding the function of interest
    :param b: a `BayesianModel` encoding the distribution of its inputs
    :param inputs: a list of variables
    :return: a scalar, Var[f]
    """

    assert isinstance(m, MarkovModel)
    assert isinstance(b, BayesianModel)

    e2x = m.get_partition_function()**2

    z = bn.util.bnmarginal(b, inputs)
    f = bn.util.divide_mms(m, z)
    f2z = bn.util.multiply_mms(f, m, inputs)
    ex2 = f2z.get_partition_function()

    return ex2 - e2x


def variance_component(m, b, inputs, i, heuristic='MinWeight'):
    """
    Compute the variance component of one of the input nodes on the expected value of an output node

    :param m: a `MarkovModel` encoding the function of interest
    :param b: a `BayesianModel` encoding the distribution of its inputs
    :param inputs: a list of strings (input variables)
    :param i: name of the node of interest
    :param heuristic: one of the elimination order heuristics supported by pgmpy. Default is 'MinWeight'
    :return: a scalar, S_i
    """

    assert isinstance(m, MarkovModel)
    assert isinstance(b, BayesianModel)
    assert all([j in m.nodes for j in inputs])
    assert all([j in b.nodes for j in inputs])
    if not isinstance(i, (list, tuple)):
        i = [i]

    # Numerator: Var_i[E_{~i}[f]]
    mminusi = bn.util.eliminate(m, to_keep=i, heuristic=heuristic, output='factor').values
    z = bn.util.bnmarginal(b, inputs)
    zminusi = bn.util.eliminate(z, to_keep=i, heuristic=heuristic, output='factor').values
    fminusi = mminusi/zminusi
    E2fminusi = np.sum(fminusi*zminusi)**2
    Efminusi2 = np.sum(fminusi**2*zminusi)

    # Denominator: Var[f]
    f = bn.util.divide_mms(m, z)
    Ef2 = bn.util.eliminate(bn.util.multiply_mms(f, m, inputs), to_keep=[], heuristic=heuristic, output='factor').values
    E2f = E2fminusi
    V = Ef2-E2f

    return (Efminusi2 - E2fminusi)/V


def total_index(m, b, inputs, i, heuristic='MinWeight'):
    """
    Compute the total index of one of the input nodes on the expected value of an output node

    We use the formula S^T_i = 1-\frac{\mathrm{Var}_{\setminus i} \left[ \ex_i[f] \right] }{\mathrm{Var}[f]}

    Saltelli, A. et al.: "Global Sensitivity Analysis: The Primer" (2008)

    :param m: a `MarkovModel` encoding the function of interest
    :param b: a `BayesianModel` encoding the distribution of its inputs
    :param inputs: a list of strings (input variables)
    :param i: name of the node of interest
    :param heuristic: one of the elimination order heuristics supported by pgmpy. Default is 'MinWeight'
    :return: a scalar, S^T_i
    """

    assert isinstance(m, MarkovModel)
    assert isinstance(b, BayesianModel)
    assert all([j in m.nodes for j in inputs])
    assert all([j in b.nodes for j in inputs])
    assert i in inputs

    # Numerator: E_i[Var_{~i}[f]]
    inputsminusi = copy.copy(inputs)
    inputsminusi.remove(i)
    mi = copy.deepcopy(m)
    bn.util.eliminate_variable(mi, i)
    z = bn.util.bnmarginal(b, inputs)
    zi = copy.deepcopy(z)
    bn.util.eliminate_variable(zi, i)
    E2fi = bn.util.eliminate(mi, to_keep=[], heuristic=heuristic, output='factor').values**2
    fi = bn.util.divide_mms(mi, zi)
    Efi2 = bn.util.eliminate(bn.util.multiply_mms(mi, fi, inputsminusi), to_keep=[], heuristic=heuristic, output='factor').values

    # Denominator: Var[f]
    f = bn.util.divide_mms(m, z)
    Ef2 = bn.util.eliminate(bn.util.multiply_mms(f, m, inputs), to_keep=[], heuristic=heuristic, output='factor').values
    E2f = E2fi
    V = Ef2-E2f

    return 1 - (Efi2 - E2fi)/V
