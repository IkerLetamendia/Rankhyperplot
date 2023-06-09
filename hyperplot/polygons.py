import numpy as np
import matplotlib.pyplot as plt
import xgi

def polygons(edges, cmap=None, nodecolors=None, nodelabels=None, ax=None, layout='pairwise_spring_layout', internode_dist=None,
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
    H = xgi.Hypergraph(edges)

    if cmap is None:
        cmap = plt.cm.Set1
    if layout == 'pairwise_spring_layout':
        pos = xgi.pairwise_spring_layout(H,)
    elif layout == 'barycenter_spring_layout':
        pos = xgi.barycenter_spring_layout(H, )
    else:
        raise ValueError("Invalid 'layout'.")

    xgi.draw(H, pos, ax=ax, cmap=cmap, nodecolors=nodecolors, nodelabels=nodelabels, **kwargs)
     

    

    

  