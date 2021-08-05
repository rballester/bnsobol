import numpy as np
import copy
import pgmpy.factors
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.inference import VariableElimination


def eliminate_variable(m, i):
    """
    Eliminates one variable of an MRF in place.

    :param m: a `pgmpy.models.MarkovModel`
    :param i: name of the node to eliminate
    """

    assert isinstance(m, MarkovModel)

    factors = [f for f in m.get_factors() if i in f.variables]
    f = pgmpy.factors.factor_product(*factors)
    pos = f.variables.index(i)
    f.values = np.sum(f.values, axis=pos)
    f.cardinality = np.delete(f.cardinality, pos)
    del f.variables[pos]
    m.add_factors(f)
    for u in range(len(f.variables)-1):
        for v in range(u+1, len(f.variables)):
            m.add_edge(f.variables[u], f.variables[v])
    m.remove_factors(*factors)
    m.remove_node(i)
    # m.check_model()


def full(m):
    """
    Compute the full factor product of all variables in the model

    :param m: an MRF (pgmpy.models.MarkovModel)
    :return: an np.ndarray with shapes according to the model's inputs
    """

    m.check_model()

    factor = m.factors[0]
    factor = factor_product(
        factor, *[m.factors[i] for i in range(1, len(m.factors))]
    )
    if set(factor.scope()) != set(m.nodes()):
        raise ValueError("DiscreteFactor for all the random variables not defined.")
    return factor


def bnmarginal(b, inputs, heuristic='MinWeight'):
    """
    Remove all non-input nodes in a Bayesian network, and returns a Markov model

    :param b: a `pgmpy.models.BayesianModel`
    :param inputs: list of nodes
    :param heuristic: one of the elimination order heuristics supported by pgmpy. Default is 'MinWeight'
    :return: a `pgmpy.models.MarkovModel`
    """

    assert isinstance(b, BayesianModel)

    # Create a copy network that includes the inputs and their ancestors, but removes non-ancestor nodes since they can be removed for free (as this is a BN)
    visited = set()
    to_visit = set(inputs)
    edges = []
    cpds = []
    while len(to_visit) > 0:
        new = to_visit.pop()
        visited.add(new)
        cpd = b.get_cpds(new)
        cpds.append(copy.deepcopy(cpd))
        for parent in cpd.variables[1:]:
            if parent not in visited and parent not in to_visit:
                to_visit.add(parent)
            edges.append([parent, new])
    b = BayesianModel(edges)
    for n in visited:
        b.add_node(n)
    b.add_cpds(*cpds)

    # Now, the most expensive part: eliminate the ancestors
    m = b.to_markov_model()
    eliminate(m, to_keep=inputs, heuristic=heuristic, output='network')
    m.check_model()
    return m


def multiply_mms(u, v, inputs):
    """
    Multiply two MRFs u and v so that, if Z(m) marginalizes an MRF m over `inputs`,
    Z(product) = Z(u)*Z(v).

    :param u: a `pgmpy.models.MarkovModel`
    :param v: a `pgmpy.models.MarkovModel`
    :return: a `pgmpy.models.MarkovModel`
    """

    assert all([i in u.nodes for i in inputs])
    assert all([i in v.nodes for i in inputs])

    def add_data(m, suffix):

        for n in list(m.nodes):
            if n not in inputs:
                nodes.add(n+suffix)

        for e in list(m.edges):
            if e[0] in inputs and e[1] in inputs:
                edges.add((e[0], e[1]))
            else:
                newedge = []
                for n in e:
                    if n in inputs:
                        newedge.append(n)
                    else:
                        newedge.append(n+suffix)
                edges.add((newedge[0], newedge[1]))

        for f in list(m.get_factors()):
            newvariables = []
            for n in f.variables:
                if n in inputs:
                    newvariables.append(n)
                else:
                    newvariables.append(n+suffix)
            newf = DiscreteFactor(newvariables, f.cardinality, f.values)
            factors.append(newf)

    nodes = set([i for i in inputs])
    edges = set()
    factors = []
    add_data(u, '_1')
    add_data(v, '_2')

    result = MarkovModel(edges)
    result.add_factors(*factors)
    result.check_model()
    return result


def divide_mms(u, v):
    """
    Divide a graphical model (Bayesian or Markov) by a MarkovModel (u / v)

    :param u: a `pgmpy.models.BayesianModel` or `pgmpy.models.MarkovModel`, the numerator
    :param v: a `pgmpy.models.MarkovModel`, the denominator
    :return: a `pgmpy.models.MarkovModel`
    """

    assert isinstance(v, MarkovModel)
    if isinstance(u, BayesianModel):
        u = u.to_markov_model()
    else:
        u = copy.deepcopy(u)
    for n in v.nodes:
        u.add_node(n)
    for edge in v.edges:
        u.add_edge(edge[0], edge[1])
    u_factors = {frozenset(f.variables): f for f in u.factors}
    for f in v.factors:
        if frozenset(f.variables) in u_factors:
            zeros = np.where(u_factors[frozenset(f.variables)].values == 0)
            u_factors[frozenset(f.variables)].values /= f.values
            u_factors[frozenset(f.variables)].values[zeros] = 0
        else:
            fnew = copy.deepcopy(f)
            fnew.values = 1/fnew.values
            if isinstance(fnew.values, np.float64):
                if fnew.values == float('Inf'):
                    fnew.values = 1e3
            else:
                fnew.values[fnew.values == float('Inf')] = 1e3
            u.add_factors(fnew)
    u.check_model()
    return u


def eliminate(m, to_keep=None, to_remove=None, output='network', heuristic='MinWeight'):
    """
    Given an MRF, eliminate a set of nodes. This does not modify the original MRF

    :param m: a `pgmpy.models.MarkovModel`
    :param to_keep: a list of nodes to remove. Pass either this or `to_remove`
    :param to_remove: a list of nodes to remove. Pass either this or `to_keep`
    :param output: 'network' (default) or 'factor'
    :param heuristic: variable ordering heuristic. Currently supported are 'MinWeight' (default) and 'MinNeighbors'
    :return: a `pgmpy.models.MarkovModel` if `output` is 'network', `pgmpy.factors.discrete.DiscreteFactor` if it is 'factor'

    TODO: MinFill, MinWeight,
    """

    if to_remove is None:
        to_remove = set(m.nodes).difference(set(to_keep))

    def cost(m, n, heuristic):
        if heuristic == 'MinNeighbors':
            return len(list(m.neighbors(n)))
        elif heuristic == 'MinWeight':
            return np.prod([m.get_cardinality(neighbor) for neighbor in m.neighbors(n)])
        else:
            raise ValueError

    if isinstance(m, BayesianModel):
        m = m.to_markov_model()
    else:
        m = m.copy()

    while len(to_remove) > 0:
        scores = {n: cost(m, n, heuristic) for n in to_remove}
        min_score_node = min(scores, key=scores.get)
        to_remove.remove(min_score_node)
        eliminate_variable(m, min_score_node)
    if output == 'network':
        return m
    else:
        f = full(m)
        f.values = f.values.transpose(*[f.variables.index(v) for v in to_keep])
        f.variables = to_keep
        return f


def add_function_node(b, outputs, function, label='function'):
    """
    Modifies a Bayesian network in place by adding a new node that is a function of a few input nodes.

    This is useful to study functions of interest that are defined in terms of a number of nodes in the network.

    :param b: a `BayesianModel`
    :param outputs: a list of nodes that `function` will depend on
    :param function: a function that takes `outputs` as arguments and returns a scalar
    """

    import inspect
    names = [b.get_cpds(output).state_names[output] for output in outputs]

    shape = [b.get_cardinality(output) for output in outputs]
    values = np.zeros(shape)
    idx = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T
    for i in range(idx.shape[0]):
        values[tuple(idx[i, :])] = function(*[names[j][idx[i, j]] for j in range(idx.shape[1])])
    values = values.reshape(1, -1)
    cpd = pgmpy.factors.discrete.CPD.TabularCPD(label, 1, evidence=outputs, evidence_card=shape, values=values)
    b.add_node(node=label)
    for output in outputs:
        b.add_edge(output, label)
    b.add_cpds(cpd)


def to_mrf(b, output, values):
    """
    Generate an MRF that encodes the expected value of an output node in a given BN. This
    is done by adding a new potential phi whose value for each O=o is o.

    :param b: A Bayesian network (pgmpy.models.BayesianModel)
    :param output: the target node
    :param values: a vector containing the values of variable `output`
    :return: an MRF (pgmpy.models.MarkovModel)
    """

    assert output in b.nodes()
    assert 'output' not in b.nodes()

    m = b.to_markov_model()
    m.add_node('output')
    m.add_edge(output, 'output')
    m.add_factors(DiscreteFactor([output, 'output'], [m.get_cardinality(output), 1], np.array(values)[:, None]))
    m.check_model()
    return m
