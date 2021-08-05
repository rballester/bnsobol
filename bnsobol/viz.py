import bnsobol as bn
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib
import matplotlib.pyplot as plt


def plot_graph(b, inputs, indices=None, output=None, filename=None):
    """
    Show a network and the sensitivity of an output with respect to a list of indices.

    :param b: a Bayesian network (`pgmpy.models.BayesianModel`)
    :param inputs: a list of nodes
    :param indices: a list of numbers (like sensitivity indices) to be shown under each input. If None (default), no numbers will be shown
    :param output: a node. If None (default), no output will be shown
    :param filename: where to save the plot. If None (default), it will be displayed on screen
    """

    labels = dict()
    nodecolors = {}
    count = 0
    if indices is not None:
        indices_rescaled = np.array(indices) / max(indices) / 2
    cmap = matplotlib.cm.get_cmap('YlGn')
    for n in b.nodes:
        labels[n] = n
        if n in inputs:
            if indices is None:
                nodecolors[n] = 'lightgreen'
            else:
                labels[n] += '\n{:.3f}'.format(indices[count])
                nodecolors[n] = cmap(indices_rescaled[count])
            count += 1
        elif n == output:
            nodecolors[n] = 'coral'
        else:
            nodecolors[n] = 'lightgray'
    plt.figure(figsize=(10, 6))
    pos = graphviz_layout(b, prog='dot')
    nx.draw_networkx_labels(b, pos=pos, font_size=8, labels=labels)
    nx.draw_networkx_nodes(b, pos=pos, node_size=500, node_color=[nodecolors[n] for n in b.nodes])
    nx.draw_networkx_edges(b, pos=pos)
    plt.axis('off')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
