import hypernetx as hnx
import matplotlib.pyplot as plt

def two_rows(edges, nodelabels=None, nodecolors=None, nodeorder=None, ax=None, nodesize=0.1, column_spacing=1):
    '''
    Plots hypergraph using bipartite two row layout, where in the bottom row contains the
    nodes and the top row the edges.
    Parameters
    ----------
    decomposed_edges : dict ({order : list of edges})
        dictionary with list of edges for each multiplet order
    nodelabels : dict
        dictionary from node to node label
    nodecolors : dict
        dictionary from node to color
    nodeorder
    nodesize : float
    column_spacing : float
    subplot_width : int
    subplot_height : int
    Returns
    -------
    '''

    if nodelabels is not None:
        nodecolors = {nodelabels[node]: color for node, color in nodecolors.items()}

    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]
        
    H = hnx.Hypergraph(edges)
    pairs = [(node, edge.uid) for edge in H.edges() for node in edge] # list of (node, edge_id) pairs
    pairscolor=[0]*len(pairs)
    for b in range(0,len(pairs)):
              [A,B]=pairs[b]
              if  A  in ['3PT-R']:
                   pairscolor[b]= 'red'
              else:
                  pairscolor[b]= 'blue'
 #   H_restrict_nodes = H.restrict_to_nodes([ '24PD', '3PT-R', '6PD'])
    hnx.drawing.two_column.draw(H, 
                                with_node_labels=True,
                                with_edge_labels=True,
                                with_color=False,
                                ax=ax,edge_kwargs={
                                    'color': pairscolor
                                },

                                #column_spacing=column_spacing,
                                #flip_orientation=True,
                                
                                #edgecolor='tab:blue',
                                #nodecolor=nodecolors,
                                #odeorder=nodeorder,
                               # nodesize=nodesize
                               )
