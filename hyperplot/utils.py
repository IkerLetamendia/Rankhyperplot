import networkx as nx
from collections import defaultdict

def create_decomposed_hypergraph(nodes, decomposed_edges):
    '''
    Creates a dict of hypergraph for each order.

    Parameters
    ----------
    nodes : list
        node names present in edges

    decomposed_edges : dict ({order : list of edges})
        dictionary with list of edges for each multiplet order, where
        each edge is a tuple of nodes

    Returns
    -------

    '''
    return {order: create_hypergraph(nodes, edges) for order, edges in decomposed_edges.items()}

def create_hypergraph(nodes, edges, remove_isolated_nodes=True):
    '''
    Creates an hypergraph.

    Parameters
    ----------
    nodes : list (e.g. [1,2,3,4,5])
        node names present in edges
    edges : list of edges (tuple of nodes) (e.g. [(2,3,4), (3,4,5), (1,2)])

    Returns
    -------
    nx.DiGraph
    '''
    g = nx.DiGraph()
    g.add_nodes_from(nodes)

    for edge in edges:
        g.add_node(tuple(edge))
        for node in edge:
            #             g.add_edge(node, tuple(edge), red=data['sorted_red'][edge_order])
            g.add_edge(node, tuple(edge))

    if remove_isolated_nodes:
        connected_nodes = set([edge[0] for edge in g.edges])
        unconnected_nodes = list(set(nodes) - connected_nodes)
        g.connected_nodes = list(connected_nodes)
        g.unconnected_nodes = list(unconnected_nodes)
        g.remove_nodes_from(unconnected_nodes)

    return g

def decompose_edges_by_len(hypergraph):
    decomposed_edges = defaultdict(list)
    for edge in hypergraph['edges']:
        decomposed_edges[len(edge)].append(edge)
    decomposition = {
        'nodes': hypergraph['nodes'],
        'edges': decomposed_edges
    }
    return decomposition