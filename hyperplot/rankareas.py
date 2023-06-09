import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt

def rankareas(edges,RSnodes,RSedges,select_EdgeorNode,
          nodelabels=None,
          nodecolors=None,
          edgecolors=None,
          ax=None,data_norm=0,
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
    edgelabels = data_norm       
    if nodelabels is not None and nodecolors is not None:
        print(nodecolors)
        nodecolors = {nodelabels[node]: color for node, color in nodecolors.items()}

    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]
        edgevals= [data_norm[edge] for edge in data_norm]
        edgelabels = [tuple([nodelabels[e] for e in edge]) for edge in data_norm]
        if  RSedges is not None and select_EdgeorNode is 'Edge' :
           RSedges = [tuple([nodelabels[e] for e in edge]) for edge in RSedges]

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
        print(nodecolors,edgecolors)

    if isinstance(nodecolors, dict):
        nodes = list(H.nodes)
        nodecolor_list = np.array([nodecolors[node] for node in nodes])
        # nodes = list(H.nodes)
        # nodecol_nodes = np.array([nodecolor[labelslist.index(nodes[node])] for node in range(len(nodes))])

    HD=H.dual()

    if select_EdgeorNode is 'Node' and RSnodes is not None:
        
       hnx.drawing.draw(HD,
                  label_alpha=0,
                  with_edge_labels=True,
                  with_node_labels=True,
                  nodes_kwargs={ 
                      'facecolors': nodecolor_list
                  },
                  edges_kwargs={ 'edgecolors': {v: 'red' if v in RSnodes else 'gray' for v in H},
                  'linewidths': {v: 6 if v in RSnodes  else 1 for v in H}},
                  node_labels_kwargs={
                      'fontsize': 14,
                  },
                  ax=ax)
       
    if select_EdgeorNode is 'Edge' and RSedges is not None:
         width=[0.2]*len(H.edges)
         for a in range(0,len(edgelabels)): 
                   [m]= edgevals[a]
                  
                   width[a]=(5*m)+0.2
                   
         edges_kwargs=dict(edgecolors=edgecolors, linewidth=width)
         numb =[ f'{a}' for a in range(0,len(RSedges))]
         H_restrict_edges = H.restrict_to_edges(numb)
         print(edges_kwargs )
         hnx.drawing.draw(H_restrict_edges,
                      label_alpha=0,
                      with_edge_labels=False,
                      with_node_labels=True,edges_kwargs=edges_kwargs, 
                      nodes_kwargs={ 
                          'facecolors': nodecolors,
                      },                      node_labels_kwargs={
                          'fontsize': 14,
                      },
                      ax=ax)

