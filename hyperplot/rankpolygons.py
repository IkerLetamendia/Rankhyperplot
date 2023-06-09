import colorsys
import numpy as np
import matplotlib.pyplot as plt
import xgi
import matplotlib.colors
def rankpolygons(edges,RSnodes,RSedges,select_EdgeorNode,data_norm=0, cmap=None, nodecolors=None, nodelabels=None, ax=None, layout='pairwise_spring_layout', internode_dist=None,
             **kwargs):
    '''
    Plots hypergraph using XGI (https://github.com/ComplexGroupInteractions/xgi/).

    Parameters
    ----------
    edges : list of with edges
    nodelabels : None, True (use node indices) or dict (node : label)
    ax : axis to plot
    layout : 'pairwise_spring_layout' or 'barycenter_spring_layout'
    internode_dist : float (1/np.sqrt(n_nodes) is the default optimal distance)
    kwargs : node_size, nodelabel_xoffset, but see xgi.draw()
    '''
    #sortsynapp=sorted(ranking,reverse=True)
    '''
    n_high=round(prvariables*len(ranking)/100)
    idx = np.argpartition(ranking, -n_high)[-n_high:]
    sortrank=sorted(ranking)
    if sortrank[len(ranking)-1]<=0 and sortrank[0]  <=0:  
        idx = np.argpartition(ranking, n_high)[:n_high] 
    RSnodes=list(idx+1)
    '''
    edgelabels = data_norm
    H = xgi.Hypergraph(edges)
    if nodelabels is not None:
        nodecolors = {nodelabels[node]: color for node, color in nodecolors.items()}
    if nodelabels is not None:
        # refactors edge names in 'decomposed_edges' and 'nodecolor' using 'nodelabels'
        edges = [tuple([nodelabels[e] for e in edge]) for edge in edges]
        edgevals= [data_norm[edge] for edge in data_norm]
        edgelabels = [tuple([nodelabels[e] for e in edge]) for edge in data_norm]
        H = xgi.Hypergraph(edges)
        if  RSedges is not None and select_EdgeorNode is 'Edge' :
           RSedges = [tuple([nodelabels[e] for e in edge]) for edge in RSedges]
           H = xgi.Hypergraph(RSedges)
  
    if cmap is None:
         cmap = plt.cm.Reds
    if layout == 'pairwise_spring_layout':
        pos = xgi.pairwise_spring_layout(H, )
    elif layout == 'barycenter_spring_layout':
        pos = xgi.barycenter_spring_layout(H, k=internode_dist)
    else:
        raise ValueError("Invalid 'layout'.")
           
    if select_EdgeorNode is 'Node' and RSnodes is not None:
      i=len(edges)
      j=len(edges[0])
      rankedges=[0]*i
      for a in range(0,len(RSnodes)): 
                    for n in range(0,i):                  
                          for m in range(0,j):  
                            if RSnodes[a]==edges[n][m]:   
                                     rankedges[n]=  1                                                                                            
      for a in range(0,len(edges)):  
            if 1==rankedges[a]:
                rankedges[a]='green'
            else:
                rankedges[a]='red'
      xgi.draw(H, pos, ax=ax,  node_labels=True,edge_fc =rankedges       , 
  dyad_lw = 4,
  node_fc = "black",
  node_ec = "green",
  node_lw = 10,
   **kwargs)
      
      
    if select_EdgeorNode is 'Edge' and RSedges is not None:
      i=len(edges)
      j=len(edges[0])
      rankedges=[0]*i
      for a in range(0,len(RSedges)): 
          for n in range(0,i):  
               if RSedges[a]==edges[n]: 
                   rankedges[n]=  1
                                    
      for a in range(0,len(edges)):  
            if 1==rankedges[a]:
                [val]=edgevals[a]
                val=int((val+0.1)*2)
                print(edgevals[a],edgevals,rankedges,val)
                rankedges[a]='#ff00%01x0' % (val)
            else:
                rankedges[a]='white'
                
      xgi.draw(H, pos, ax=ax,  node_labels=True,edge_fc =rankedges, 
    dyad_lw = 4,
    node_fc = "black",
    node_ec = "green",
    node_lw = 10,
     **kwargs)


     

    

    

  