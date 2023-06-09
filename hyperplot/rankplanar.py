import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import NetworkXException

import hyperplot.utils

def rankplanar(edges,RSnodes,RSedges,select_EdgeorNode='Edge', nodelabels=None, ax=None,data_norm=0):
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
    edgelabels = data_norm

    if nodelabels is not None:
        nodes = [nodelabels[n] for n in nodes]
        
    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]
        edgevals= [data_norm[edge] for edge in data_norm]
        edgelabels = [tuple([nodelabels[e] for e in edge]) for edge in data_norm]
        if  RSedges is not None and select_EdgeorNode is 'Edge' :
          RSedges = [tuple([nodelabels[e] for e in edge]) for edge in RSedges]
         
  
    g = hyperplot.utils.create_hypergraph(nodes, edges, remove_isolated_nodes=True)
    # I like planar layout, but it cannot be used in general
    try:
        pos = nx.planar_layout(g)
    except NetworkXException:
        print('Failed using planar_layout, using spring_layout instead.')
        pos = nx.spring_layout(g)
       # pos = nx.kamada_kawai_layout(g)
        
    # Plot true nodes in orange, star-expansion edges in red
    nodes = g.connected_nodes
    extra_nodes = set(g.nodes) - set(g.connected_nodes)
    
    if select_EdgeorNode is 'Node' and RSnodes is not None:
        rankedges=[0]*len(g.edges) 
        width=[0]*len(g.edges)
        for n, val in enumerate(g.edges):   
           for m in range(0,len(edges[0])):  
               for a in range(0,len(RSnodes)): 
                                  [nval,nedge]=val
                                  if RSnodes[a]==nval:   
                                     rankedges[n]=1           
        for a in range(0,len(g.edges)):  
            if 1==rankedges[a]:
                rankedges[a]='green'
                width[a]=5
            else:
                rankedges[a]='blue'
                width[a]=0.1
        node_color= ['red' if v in RSnodes else 'orange' for v in nodes]
        nodesz=[50]*len(nodes)    
        for b in range(0,len(nodes)):
                  A=nodes[b]
                  if A in RSnodes :
                       nodesz[b]= 100
                  else:
                      nodesz[b]= 30

        nx.draw_networkx_nodes(g, pos, node_size=nodesz, nodelist=nodes,
                               ax=ax, node_color=node_color)
        nx.draw_networkx_nodes(g, pos, node_size=100, nodelist=extra_nodes,
                               ax=ax, node_color='#3486eb')
        nx.draw_networkx_edges(g, pos, ax=ax,edge_color=rankedges,width=width,
                       connectionstyle='arc3,rad=0.05', arrowstyle='-')

        # Draw labels only for true nodes
        labels = {node: str(node) for node in nodes}
        for node in nodes:
          [x,y]=pos[node]
          pos[node]=[x+0.04,y]
        nx.draw_networkx_labels(g, pos, labels, ax=ax, horizontalalignment="left",)
        
    if select_EdgeorNode is 'Edge':  
        rankedges=[0]*len(g.edges)
        ranknodes=[0]*len(g.nodes)
        width=[0.2]*len(g.edges)
        if RSedges is not None:
          for a in range(0,len(RSedges)): 
            for n, val in enumerate(g.edges):   
                [nval,nedge]=val
                if RSedges[a]==nedge:
                    rankedges[n]=1
                    f=0
                    for f in range(0,len(g.nodes)):
                     if ranknodes[f]==nval:
                         break
                     if ranknodes[f]==0:
                         ranknodes[f]=nval
                         break

        if edges is not None:             
         for a in range(0,len(edgelabels)): 
           for n, val in enumerate(g.edges):   
               [nval,nedge]=val
               if nedge==edgelabels[a]:
                   [m]= edgevals[a]
                   width[n]=(3*m)+0.2                       
               if 1==rankedges[n]:
                 rankedges[n]='green'
               if 0==rankedges[n]:
                 rankedges[n]='blue'
         
        node_color= ['red' if v in ranknodes else 'orange' for v in nodes]        
        nodesz=[ 50 if v in ranknodes else 10 for v in nodes]  
        nx.draw_networkx_nodes(g, pos, node_size=nodesz, nodelist=nodes,
                               ax=ax, node_color=node_color)
        nx.draw_networkx_nodes(g, pos, node_size=100, nodelist=extra_nodes,
                               ax=ax, node_color='#3486eb') #
        nx.draw_networkx_edges(g, pos, ax=ax,edge_color=rankedges,width=width,
                       connectionstyle='arc3,rad=0.05', arrowstyle='-')

        ranknodes=[node for node in ranknodes if node !=0]
        # Draw labels only for true nodes
        labels = {node: str(node) for node in ranknodes}
      
        for node in nodes:
          [x,y]=pos[node]
         
          pos[node]=[x+0.04,y]
        nx.draw_networkx_labels(g, pos, labels, ax=ax,font_size=12, horizontalalignment="left",)

