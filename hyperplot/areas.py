import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt

def areas(edges,
          nodelabels=None,
          nodecolors=None,
          edgecolors=None,
          ax=None,
          linewidth=1):

    '''
    Plots hypergraph where nodes are points where each edges is an area
    encircling nodes in edge.
    An edge is a k-tuple of nodes (e.g. (2,4,6)) where k is the edge order.
    Parameters
    ----------
    decomposed_edges : dict ({order : list of edges})
        dictionary with list of edges for each multiplet order
    nodelabels : dict
        dictionary from node to node label
    nodecolors : dict
        dictionary from node to color
    edgecolors : color string or dict
        either a color for all edges, or a map from edges to color.       
    linewidth : float
        line width of areas.
    '''

    if nodelabels  is not None:
        nodecolors = {nodelabels[node]: color for node, color in nodecolors.items()}

    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]

    H = hnx.Hypergraph(edges)

    # get vals
    if edgecolors is None:
        edges_kwargs = {}
    elif isinstance(edgecolors, dict):
        cmap = plt.cm.viridis
        alpha = .8
        edge_elements = [tuple(H.edges[edge].elements) for edge in H.edges]
        vals = np.array([edgecolors[e] for e in edge_elements])
        norm = plt.Normalize(vals.min(), vals.max())
        edgecolors = cmap(norm(vals)) * (1, 1, 1, alpha)
        edges_kwargs = dict(edgecolors=edgecolors, linewidth=linewidth)
    else:
        edges_kwargs = dict(edgecolors=edgecolors, linewidth=linewidth)

    if nodecolors is None:
        nodecolors = 'black'

    if isinstance(nodecolors, dict):
        nodes = list(H.nodes)
        nodecolor_list = np.array([nodecolors[node] for node in nodes])
        # nodes = list(H.nodes)
        # nodecol_nodes = np.array([nodecolor[labelslist.index(nodes[node])] for node in range(len(nodes))])
  #  H_restrict_nodes = H.collapse_nodes()
    hnx.drawing.draw(H,
                     label_alpha=0,
                     with_edge_labels=False,
                     with_node_labels=True,
                     nodes_kwargs={
                         'facecolors': nodecolor_list
                     },
                     edges_kwargs=edges_kwargs,
                     node_labels_kwargs={
                         'fontsize': 14,
                     },
                     ax=ax)