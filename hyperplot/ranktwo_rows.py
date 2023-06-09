import networkx as nx
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt

def ranktwo_rows(edges,RSnodes,RSedges,select_EdgeorNode, nodelabels=None, nodecolors=None, nodeorder=None, ax=None, nodesize=0.5, column_spacing=1,data_norm=0):
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
    edgelabels = data_norm
    if nodelabels is not None:
        nodecolors = {nodelabels[node]: color for node, color in nodecolors.items()}
    
    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
       edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]
       edgevals= [data_norm[edge] for edge in data_norm]
       edgelabels = [tuple([nodelabels[e] for e in edge]) for edge in data_norm]
       if  RSedges is not None and select_EdgeorNode is 'Edge' :
          RSedges = [tuple([nodelabels[e] for e in edge]) for edge in RSedges]

    H = hnx.Hypergraph(edges)
    pairs = [(node, edge.uid) for edge in H.edges() for node in edge] # list of (node, edge_id) pairs
   
    if select_EdgeorNode is 'Node' and RSnodes is not None:
        width=[0]*len(pairs)
        pairscolor=[0]*len(pairs)
        for b in range(0,len(pairs)):
              [A,B]=pairs[b]
              if  A  in RSnodes:
                   pairscolor[b]= 'red'
                   width[b]=2
              else:
                  pairscolor[b]= 'blue'
                  width[b]=0.1
       
        hnx.drawing.two_column.draw(H, 
                                    with_node_labels=True,
                                    with_edge_labels=True,
                                    with_color=False,
                                    ax=ax,edge_kwargs={'color':pairscolor,'linewidth':width})
        
    if select_EdgeorNode is 'Edge':

         width=[0.2]*len(pairs)
         pairscolor=['blue']*len(pairs)
         
         x='0'
         y=0
         if edges is not None: 
          for b in range(0,len(pairs)):
                [A,B]=pairs[b]  
            
                if B==x:
                    [m]= edgevals[int(x)]
                    width[y]=(5*m)+0.2
                    y=y+1
                if B!=x:
                    x=B
                   # print(m)
                    [m]= edgevals[int(x)]
                    width[y]=(5*m)+0.2
                    y=y+1
                x=B

         if RSedges is not None:
          for b in range(0,len(pairs)):
              [A,B]=pairs[b]
              if  int(B)<len(RSedges):
                   pairscolor[b]= 'red'
                  # width[b]=2
              else:
                  pairscolor[b]=  'blue'
                 # width[b]=0.2

         hnx.drawing.two_column.draw(H, 
                                    with_node_labels=True,
                                    with_edge_labels=True,
                                    with_color=False,
                                    ax=ax
                                  ,edge_kwargs={'color':pairscolor,'linewidth':width})