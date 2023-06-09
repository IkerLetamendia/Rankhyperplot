import matplotlib.pyplot as plt
import networkx as nx
from networkx import NetworkXException

import hyperplot.utils

def planar(edges, nodelabels=None, ax=None):
    '''
    Plots hypergraph using random planar layout.
    Parameters
    ----------
    decomposed_edges : dict ({order : list of edges})
        dictionary with list of edges for each multiplet order
    nodelabels : dict
        dictionary from node to node label
    '''

    nodes = list(set(sum([list(e) for e in edges], [])))

    if nodelabels is not None:
        nodes = [nodelabels[n] for n in nodes]
    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]

    g = hyperplot.utils.create_hypergraph(nodes, edges, remove_isolated_nodes=True)

    # I like planar layout, but it cannot be used in general
    try:
        pos = nx.planar_layout(g)
    except NetworkXException:
        print('Failed using planar_layout, using spring_layout instead.')
        pos = nx.spring_layout(g)

    # Plot true nodes in orange, star-expansion edges in red
    nodes = g.connected_nodes
    extra_nodes = set(g.nodes) - set(g.connected_nodes)
    node_color=['red' if v in ['26FS','23FS','24PD','3PT-R'] else '#3486eb' for v in nodes]
    node_size=[400 if v in ['26FS','23FS','24PD','3PT-R'] else 200 for v in nodes]
    nx.draw_networkx_nodes(g, pos, node_size=200, nodelist=nodes,
                          ax=ax, node_color='#f77f00')
    #nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=nodes,
     #                      ax=ax,node_color=node_color )
    
    nx.draw_networkx_nodes(g, pos, node_size=100, nodelist=extra_nodes,
                           ax=ax, node_color='#3486eb')
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color= node_color,
                           connectionstyle='arc3,rad=0.05', arrowstyle='-')

    # Draw labels only for true nodes
    labels = {node: str(node) for node in nodes}
    nx.draw_networkx_labels(g, pos, labels, ax=ax)
