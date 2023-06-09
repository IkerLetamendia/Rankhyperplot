


"""Draw hypergraphs and simplicial complexes with matplotlib."""

from collections.abc import Iterable
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .. import convert
from ..classes import Hypergraph, SimplicialComplex, max_edge_order
from ..exception import XGIError
from ..stats import EdgeStat, NodeStat
from .layout import barycenter_spring_layout

__all__ = [
    "draw",
]


def draw(
    H,
    pos=None,
    ax=None,
    dyad_color="black",
    dyad_lw=1.5,
    edge_fc=None,
    node_fc="white",
    node_ec="black",
    node_lw=1,
    node_size=10,
    max_order=None, 
    **kwargs,
):
    """Draw hypergraph or simplicial complex.

    Parameters
    ----
    H : Hypergraph or SimplicialComplex.

    pos : dict (default=None)
        If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
        If None (default), use the `barycenter_spring_layout` to compute the positions.

    ax : matplotlib.pyplot.axes (default=None)

    dyad_color : color (str, dict, or iterable, default='black')
        Color of the dyadic links.  If str, use the
        same color for all edges.  If a dict, must contain (edge_id: color_str) pairs.  If
        iterable, assume the colors are specified in the same order as the edges are found
        in H.edges.

    dyad_lw :  float (default=1.5)
        Line width of edges of order 1 (dyadic links).

    edge_fc : str, 4-tuple, ListedColormap, LinearSegmentedColormap, or dict of 4-tuples or strings
        Color of hyperedges

    node_fc : color (str, dict, or iterable, default='white')
        Color of the nodes.  If str, use the same color for all nodes.  If a dict, must
        contain (node_id: color_str) pairs.  If other iterable, assume the colors are
        specified in the same order as the nodes are found in H.nodes.

    node_ec : color (dict or str, default='black')
        Color of node borders.  If str, use the same color for all nodes.  If a dict, must
        contain (node_id: color_str) pairs.  If other iterable, assume the colors are
        specified in the same order as the nodes are found in H.nodes.

    node_lw : float (default=1.0)
        Line width of the node borders in pixels.

    node_size : float (default=0.03)
        Radius of the nodes in pixels

    max_order : int, optional
        Maximum of hyperedges to plot. Default is None (plot all orders).

    **kwargs : optional args
        alternate default values. Values that can be overwritten are the following:
        * min_node_size
        * max_node_size
        * min_dyad_lw
        * max_dyad_lw
        * min_node_lw
        * max_node_lw
        * node_fc_cmap
        * node_ec_cmap
        * edge_fc_cmap
        * dyad_color_cmap

    Examples
    --------
    >>> import xgi
    >>> H = xgi.Hypergraph()
    >>> H.add_edges_from([[1,2,3],[3,4],[4,5,6,7],[7,8,9,10,11]])
    >>> xgi.draw(H, pos=xgi.barycenter_spring_layout(H))

    """
    settings = {
        "min_node_size": 10,
        "max_node_size": 30,
        "min_dyad_lw": 2,
        "max_dyad_lw": 10,
        "min_node_lw": 1,
        "max_node_lw": 5,
        "node_fc_cmap": cm.Reds,
        "node_ec_cmap": cm.Greys,
        "edge_fc_cmap": cm.Blues,
        "dyad_color_cmap": cm.Greys,
    }

    settings.update(kwargs)

    if edge_fc is None:
        edge_fc = H.edges.size

    if pos is None:
        pos = barycenter_spring_layout(H)

    if ax is None:
        ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis("off")

    if not max_order:
        max_order = max_edge_order(H)

    if isinstance(H, SimplicialComplex):
        draw_xgi_simplices(
            H, pos, ax, dyad_color, dyad_lw, edge_fc, max_order, settings
        )
    elif isinstance(H, Hypergraph):
        draw_xgi_hyperedges(
            H, pos, ax, dyad_color, dyad_lw, edge_fc, max_order, settings
        )
    else:
        raise XGIError("The input must be a SimplicialComplex or Hypergraph")

    draw_xgi_nodes(
        H,
        pos,
        ax,
        node_fc,
        node_ec,
        node_lw,
        node_size,
        max_order,
        settings,
    )


def draw_xgi_nodes(
    H,
    pos,
    ax,
    node_fc,
    node_ec,
    node_lw,
    node_size,
    zorder,
    settings,
):
    """Draw the nodes of a hypergraph

    Parameters
    ----------
    ax : axis handle
        Plot axes on which to draw the visualization
    H : Hypergraph or SimplicialComplex
        Higher-order network to plot
    pos : dict of lists
        The x, y position of every node
    node_fc : str, 4-tuple, or dict of strings or 4-tuples
        The face color of the nodes
    node_ec : str, 4-tuple, or dict of strings or 4-tuples
        The outline color of the nodes
    node_lw : int, float, or dict of ints or floats
        The line weight of the outline of the nodes
    node_size : int, float, or dict of ints or floats
        The node radius
    zorder : int
        The layer on which to draw the nodes
    settings : dict
        Default parameters. Keys that may be useful to override default settings:
        * node_fc_cmap
        * node_ec_cmap
        * min_node_lw
        * max_node_lw
        * min_node_size
        * max_node_size
    """
    # Note Iterable covers lists, tuples, ranges, generators, np.ndarrays, etc
    node_fc = _color_arg_to_dict(node_fc, H.nodes, settings["node_fc_cmap"])
    node_ec = _color_arg_to_dict(node_ec, H.nodes, settings["node_ec_cmap"])
    node_lw = _scalar_arg_to_dict(
        node_lw,
        H.nodes,
        settings["min_node_lw"],
        settings["max_node_lw"],
    )
    node_size = _scalar_arg_to_dict(
        node_size, H.nodes, settings["min_node_size"], settings["max_node_size"]
    )

    for i in H.nodes:
        (x, y) = pos[i]
        ax.scatter(
            x,
            y,
            s=node_size[i] ** 2,
            c=node_fc[i],
            edgecolors=node_ec[i],
            linewidths=node_lw[i],
            zorder=zorder,
        )


def draw_xgi_hyperedges(H, pos, ax, dyad_color, dyad_lw, edge_fc, max_order, settings):
    """Draw hyperedges.

    Parameters
    ----------
    ax : axis handle
        Figure axes to plot onto
    H : Hypergraph
        A hypergraph
    pos : dict of lists
        x,y position of every node
    dyad_color : str, 4-tuple, or dict of 4-tuples or strings
        The color of the pairwise edges
    dyad_lw : int, float, or dict
        Pairwise edge widths
    edge_fc : str, 4-tuple, ListedColormap, LinearSegmentedColormap, or dict of 4-tuples or strings
        Color of hyperedges
    max_order : int, optional
        Maximum of hyperedges to plot. Default is None (plot all orders).
    settings : dict
        Default parameters. Keys that may be useful to override default settings:
        * dyad_color_cmap
        * min_dyad_lw
        * max_dyad_lw
        * edge_fc_cmap

    Raises
    ------
    XGIError
        If a SimplicialComplex is passed.
    """
    dyad_color = _color_arg_to_dict(dyad_color, H.edges, settings["dyad_color_cmap"])
    dyad_lw = _scalar_arg_to_dict(
        dyad_lw, H.edges, settings["min_dyad_lw"], settings["max_dyad_lw"]
    )

    edge_fc = _color_arg_to_dict(edge_fc, H.edges, settings["edge_fc_cmap"])
    # Looping over the hyperedges of different order (reversed) -- nodes will be plotted separately

    for id, he in H.edges.members(dtype=dict).items():
        d = len(he) - 1
        if d > max_order:
            continue

        if d == 1:
            # Drawing the edges
            he = list(he)
            x_coords = [pos[he[0]][0], pos[he[1]][0]]
            y_coords = [pos[he[0]][1], pos[he[1]][1]]
            line = plt.Line2D(
                x_coords,
                y_coords,
                color=dyad_color[id],
                lw=dyad_lw[id],
                zorder=max_order - 1,
            )
            ax.add_line(line)

        else:
            # Hyperedges of order d (d=1: links, etc.)
            # Filling the polygon
            coordinates = [[pos[n][0], pos[n][1]] for n in he]
            # Sorting the points counterclockwise (needed to have the correct filling)
            sorted_coordinates = _CCW_sort(coordinates)
            obj = plt.Polygon(
                sorted_coordinates,
                facecolor=edge_fc[id],
                alpha=0.4,
                zorder=max_order - d,
            )
            ax.add_patch(obj)


def draw_xgi_simplices(SC, pos, ax, dyad_color, dyad_lw, edge_fc, max_order, settings):
    """Draw maximal simplices and pairwise faces.

    Parameters
    ----------
    ax : axis handle
        Figure axes to plot onto
    SC : SimplicialComplex
        A simpicial complex
    pos : dict of lists
        x,y position of every node
    dyad_color : str, 4-tuple, or dict of 4-tuples or strings
        The color of the pairwise edges
    dyad_lw : int, float, or dict
        Pairwise edge widths
    edge_fc : str, 4-tuple, ListedColormap, LinearSegmentedColormap, or dict of 4-tuples or strings
        Color of simplices
    max_order : int, optional
        Maximum of hyperedges to plot. Default is None (plot all orders).
    settings : dict
        Default parameters. Keys that may be useful to override default settings:
        * dyad_color_cmap
        * min_dyad_lw
        * max_dyad_lw
        * edge_fc_cmap

    Raises
    ------
    XGIError
        If a SimplicialComplex is passed.
    """
    # I will only plot the maximal simplices, so I convert the SC to H
    H_ = convert.from_simplicial_complex_to_hypergraph(SC)

    dyad_color = _color_arg_to_dict(dyad_color, H_.edges, settings["dyad_color_cmap"])
    dyad_lw = _scalar_arg_to_dict(
        dyad_lw,
        H_.edges,
        settings["min_dyad_lw"],
        settings["max_dyad_lw"],
    )

    edge_fc = _color_arg_to_dict(edge_fc, H_.edges, settings["edge_fc_cmap"])
    # Looping over the hyperedges of different order (reversed) -- nodes will be plotted separately
    for id, he in H_.edges.members(dtype=dict).items():
        d = len(he) - 1
        if d > max_order:
            continue
        if d == 1:
            # Drawing the edges
            he = list(he)
            x_coords = [pos[he[0]][0], pos[he[1]][0]]
            y_coords = [pos[he[0]][1], pos[he[1]][1]]

            line = plt.Line2D(x_coords, y_coords, color=dyad_color[id], lw=dyad_lw[id])
            ax.add_line(line)
        else:
            # Hyperedges of order d (d=1: links, etc.)
            # Filling the polygon
            coordinates = [[pos[n][0], pos[n][1]] for n in he]
            # Sorting the points counterclockwise (needed to have the correct filling)
            sorted_coordinates = _CCW_sort(coordinates)
            obj = plt.Polygon(
                sorted_coordinates,
                facecolor=edge_fc[id],
                alpha=0.4,
            )
            ax.add_patch(obj)
            # Drawing the all the edges within
            for i, j in combinations(sorted_coordinates, 2):
                x_coords = [i[0], j[0]]
                y_coords = [i[1], j[1]]
                line = plt.Line2D(
                    x_coords, y_coords, color=dyad_color[id], lw=dyad_lw[id]
                )
                ax.add_line(line)


def _scalar_arg_to_dict(arg, ids, min_val, max_val):
    """Map different types of arguments for drawing style to a dict with scalar values.

    Parameters
    ----------
    arg : int, float, dict, iterable, or NodeStat/EdgeStat
        Attributes for drawing parameter
    ids : NodeView or EdgeView
        This is the node or edge IDs that attributes get mapped to.
    min_val : int or float
        The minimum value of the drawing parameter
    max_val : int or float
        The maximum value of the drawing parameter

    Returns
    -------
    dict
        An ID: attribute dictionary

    Raises
    ------
    TypeError
        If a int, float, list, dict, or NodeStat/EdgeStat is not passed
    """
    if isinstance(arg, dict):
        return {id: arg[id] for id in arg if id in ids}
    elif type(arg) in [int, float]:
        return {id: arg for id in ids}
    elif isinstance(arg, NodeStat) or isinstance(arg, EdgeStat):
        vals = np.interp(arg.asnumpy(), [arg.min(), arg.max()], [min_val, max_val])
        return dict(zip(ids, vals))
    elif isinstance(arg, Iterable):
        return {id: arg[idx] for idx, id in enumerate(ids)}
    else:
        raise TypeError(
            f"argument must be dict, str, or iterable, received {type(arg)}"
        )


def _color_arg_to_dict(arg, ids, cmap):
    """Map different types of arguments for drawing style to a dict with color values.

    Parameters
    ----------
    arg : str, dict, iterable, or NodeStat/EdgeStat
        Attributes for drawing parameter
    ids : NodeView or EdgeView
        This is the node or edge IDs that attributes get mapped to.
    cmap : ListedColormap or LinearSegmentedColormap
        colormap to use for NodeStat/EdgeStat

    Returns
    -------
    dict
        An ID: attribute dictionary

    Raises
    ------
    TypeError
        If a string, tuple, list, or dict is not passed
    """
    if isinstance(arg, dict):
        return {id: arg[id] for id in arg if id in ids}
    elif type(arg) in [tuple, str]:
        return {id: arg for id in ids}
    elif isinstance(arg, NodeStat) or isinstance(arg, EdgeStat):
        if isinstance(cmap, ListedColormap):
            vals = np.interp(arg.asnumpy(), [arg.min(), arg.max()], [0, cmap.N])
        elif isinstance(cmap, LinearSegmentedColormap):
            vals = np.interp(arg.asnumpy(), [arg.min(), arg.max()], [0.1, 0.9])
        else:
            raise XGIError("Invalid colormap!")

        return {id: np.array(cmap(vals[i])).reshape(1, -1) for i, id in enumerate(ids)}
    elif isinstance(arg, Iterable):
        return {id: arg[idx] for idx, id in enumerate(ids)}
    else:
        raise TypeError(
            f"argument must be dict, str, or iterable, received {type(arg)}"
        )


def _CCW_sort(p):
    """
    Sort the input 2D points counterclockwise.
    """
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


"""
hemendik aurrea ordezko analysis


"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import toolbag.toolbag.read_write
import hyperplot

from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
# IMPORTANT NOTE:
# Functions in this library load dataset by creating a 'data' structure (dict) with the following information that
# is needed to call the plotting functions (e.g. plot_polygons(data)) that wrap around Hyperplot:
#    data : dict
#         'edges' : {'red': {order (int) : list of edges}, 'syn': {order (int) : list of edges}}
#           (dictionary of edges for both redundancy and synergy, for each multiplex order)
#         'node2labels' : {node : label}, None
#         'node2colors' : {node : color}, None
#         'orders' : list of int with multiplex orders (e.g. [3, 4, 5, 6])
#
# 'data' contains more field, but the ones above are essential to call the plotting functions here.

## FUNCTIONS TO LOAD DATASETS
def rawdata2data(rawdata, min_ord=3, max_ord=6):
    '''
    Input: rawdata (dict with '__header__', 'Otot', 'data' fields).
    Output: dict with Otot fields (sorted_red, index_red, bootsig_red, etc) and 'orders'.
    '''

    data = {'sorted_red': {}, 'index_red': {}, 'bootsig_red': {},
            'sorted_syn': {}, 'index_syn': {}, 'bootsig_syn': {},}

    for order in range(min_ord, max_ord+1):
        for key in data.keys():

            tmp = rawdata['Otot'][order - 1][key]

            if not hasattr(tmp, '__len__'):  # convert matlab singletons to array
                tmp = np.array([tmp])

            if key == 'index_syn' or key == 'index_red':  # VERY IMPORTANT!! matlab to python indexing
                data[key][order] = tmp - 1
            else:
                data[key][order] = tmp  # python indexing

    data['orders'] = list(range(min_ord, max_ord+1))

    return data

def add_datainfo2data(data, datainfo):
    '''
    Adds 'data', 'n_dims' and 'n_points' fields to data.

    Valid for 'empathy', 'eating', 'PTSD' datasets only.
    '''
    print('Adding data info...')
    data_shape = datainfo.shape
    print(f"shape: {data_shape}")
    data['data'] = datainfo
    data['n_dims'] = data_shape[1]
    data['n_points'] = data_shape[0]

def add_hypergraph2data(data):
    '''
    Add 'edges', 'edge2vals', 'hypergraph', 'node' to data.
    '''
    print('Adding hypergraph info...')

    decomposed_edges, decomposed_edge2vals = get_decomposed_edge_and_vals(data)
    nodes = list(range(1, data['n_dims'] + 1))
    hypergraphs = {stat: hyperplot.utils.create_decomposed_hypergraph(nodes, decomposed_edges[stat]) for stat in ['syn', 'red']}

    data['edges'] = decomposed_edges
    data['edge2vals'] = decomposed_edge2vals
    data['nodes'] = nodes
    data['hypergraph'] = hypergraphs


def flip_color2node(color2nodes):
    '''
    Turns color2node into node2color dict.
    '''

    node2color = {}
    for color, nodes in color2nodes.items():
        for node in nodes:
            node2color[node] = color
    return node2color


def get_decomposed_edge_and_vals(data):
    '''
    PARAMETERS
    ----------
    data : {'sorted_red' : {order : vals}, 'index_red' : {order : ixs},
            'sorted_syn' : {order : vals}, 'index_syn' : {order : ixs},
            'n_dims' : int, 'n_points' : int, 'data' : (n_dims, n_points)}

    RETURNS
    -------
    decomposed_edges : {order : tuple}
    decomposed_edge2val : {order : {edge : val}}
    '''
    # process data into
    # decomposed_edges = {red/syn : {order : list of edges}}
    # decomposed_edge2vals = {red/syn : {order : {edge : val}}}

    decomposed_edges = {'red': {}, 'syn': {}}
    decomposed_edge2vals = {'red': {}, 'syn': {}}

    for stat in ['red', 'syn']:
        print(f'>>> {stat.upper()}')

        decomposed_ixs = data['index_' + stat]
        decomposed_vals = data['sorted_' + stat]

        # edges in the hypergraph
        print('Retrieving edges...')
        orders = data['index_' + stat].keys()
        for order in orders:
            n_dims = data['n_dims']

            # ATTENTION HERE: comes from matlab nchoosek function, e.g. a=nchoosek(1:53,3) (53 is n_dim, 3 is the order of multiplet)
            index2edge = list(itertools.combinations(range(1, n_dims + 1), order))

            ixs = decomposed_ixs[order]
            print(f'Order: {order} | ixs: {ixs}')
            decomposed_edges[stat][order] = [index2edge[ix] for ix in ixs]

        # values for each edge
        print('Retrieving edge values...')
        edge2vals = {}
        for order in orders:
            decomposed_edge2vals[stat][order] = {}
            edges = decomposed_edges[stat][order]
            vals = decomposed_vals[order] 
            print(f'Order: {order} | vals: {vals}')
            for edge, val in zip(edges, vals):
                decomposed_edge2vals[stat][order][edge]  = val   # add order field?
    
    return decomposed_edges, decomposed_edge2vals
     
def load_empathy_dataset(fpath):
    '''
    Load Briganti 2017 empathy dataset.
    '''
    empathy_color2nodes = {'red': [1, 5, 7, 12, 16, 23, 26],
                           'lightblue': [2, 4, 9, 14, 18, 20, 22],
                           'blue': [6, 10, 13, 17, 19, 24, 27],
                           'orange': [3, 8, 11, 15, 21, 25, 28]}
    node2colors = flip_color2node(empathy_color2nodes)

    nodes = [1, 5, 7, 12, 16, 23, 26,
             2, 4, 9, 14, 18, 20, 22,
             6, 10, 13, 17, 19, 24, 27,
             3, 8, 11, 15, 21, 25, 28]

    node2labels = {1:'1FS', 5:'5FS', 7:'7FS-R', 12:'12FS-R', 16:'16FS', 23:'23FS', 26:'26FS',
                   3:'3PT-R', 8:'8PT', 11:'11PT', 15:'15PT-R', 21:'21PT', 25:'25PT', 28:'28PT',
                   2:'2EC', 4:'4EC-R', 9:'9EC', 14:'14EC-R', 18:'18EC-R', 20:'20EC', 22:'22EC',
                   6:'6PD', 10:'10PD', 13:'13PD-R', 17:'17PD', 19:'19PD-R', 24:'24PD', 27:'27PD'
                   }

    nodeorder = {node: n for n, node in enumerate(nodes)}

    # load dataset
    data = load_dataset(fpath, min_ord=3, max_ord=5)

    data['nodeorder'] = nodeorder
    data['node2labels'] = node2labels
    data['node2labels'] = None
    data['node2colors'] = node2colors


    return data

def load_eating_dataset(fpath):
    '''
    Load Eating disorders dataset.
    '''

    labels = ['Dft', 'Bul', 'Bod', 'Ine', 'Per', 'Dis', 'Awa', 'Fea', 'Asm', 'Imp', 'Soc', 'BDI',
              'Anx', 'Res', 'Nov', 'Har', 'Red', 'Pes', 'Sed', 'Coa', 'Set', 'Dir', 'Aut', 'Lim',
              'Foc', 'Inh', 'Mis', 'Sta', 'Exp', 'Cri', 'Qua', 'Pref']

    eating_color2labels = {'#7bba72' : ['Mis', 'Qua', 'Pref', 'Sta', 'Cri', 'Exp'],
                           '#ad9a53' : ['Soc', 'Asm', 'Imp', 'Per', 'Bod', 'Dft', 'Ine', 'Bul',
                                'Dis', 'Awa', 'Fea'],
                           '#789cff' : ['Sed', 'Har', 'Pes', 'Nov', 'Coa', 'Red', 'Set'],
                           '#d78adb' : ['Aut', 'Inh', 'Dir', 'Lim', 'Foc'],
                           '#cf5540' : ['BDI'],
                           '#48c0c2' : ['Anx', 'Res']}

    label2colors = flip_color2node(eating_color2labels)
    node2colors = {labels.index(label) + 1: color for label, color in label2colors.items()}
    node2labels = {node:label for node, label in zip(range(1, len(labels)+1), labels)}

    # load dataset
    data = load_dataset(fpath, min_ord=3, max_ord=6)

    data['nodeorder'] = None
    data['node2labels'] = node2labels
    #data['node2labels'] = None
    data['node2colors'] = node2colors

    return data

def load_fmri_dataset(fpath):
    fMRI_color2nodes = {'#ffb169': [30, 41, 99, 45, 50],
                        '#7eed64': [66, 76],
                        '#348feb': [2, 5, 8, 4, 23, 97, 74, 79, 69],
                        '#a35ef2': [6, 1, 25, 13, 14, 43, 19, 7, 98],
                        '#e65555': [71, 65, 42, 93, 53, 83, 75, 31, 90, 78, 81, 95, 73, 70, 54, 96, 27],
                        '#d1d1d1': [20, 35, 29, 52, 34, 24, 85],
                        '#94fff6': [48, 77, 26, 88]}

    node2colors = flip_color2node(fMRI_color2nodes)

    data = load_dataset(fpath, min_ord=3, max_ord=6, n_dims=53)

    data['nodeorder'] = None
    data['node2labels'] = None
    data['node2colors'] = node2colors

    return data


def load_dataset(fpath, min_ord, max_ord, n_dims=None, n_points=None):
    '''
    Load dataset (output from O-info analysis, i.e. Otot structure)

    Returns
    -------
    data : dict ('sorted_red', 'index_red', 'bootsig_red', 'sorted_syn', 'index_syn', 'bootsig_syn')
    '''
    rawdata = toolbag.toolbag.read_write.loadmat(fpath)
    rawdata['Otot'] = [toolbag.toolbag.read_write._todict(x) for x in rawdata['Otot']]
    data = rawdata2data(rawdata, min_ord, max_ord)
    if 'data' in rawdata.keys():
        add_datainfo2data(data, rawdata['data'])
    else:
        data['n_points'] = n_points
        data['n_dims'] = n_dims
    add_hypergraph2data(data)
    return data

## FUNCTIONS TO PLOT O-INFO ANALYSIS USING HYPERPLOT

def plot_polygons(data, internode_dists=[None, None], show_nodelabels=True, **kwargs):
    '''
    Plot O-info hypergraph using polygons

    Parameters
    ----------
    data : dict
        'edges' : {'red': {order (int) : list of edges}, 'syn': {order (int) : list of edges}}
        'node2labels' : {node : label}
        'node2colors' : {node : color}
        'orders' : list of int with multiplex orders (e.g. [3, 4, 5, 6])

    internode_dists : [float, float], where 1/np.sqrt(n_nodes) is the default optimal distance
    show_nodelabels : bool
    kwargs : node_size, nodelabel_xoffset, but see xgi.draw()
    '''

    n_plots = len(data['orders'])

    n_nodes = len(data['nodes'])
    k_opt = 1 / np.sqrt(n_nodes)
    print(f"Optimal internode distance: {k_opt:.2f}")

    if show_nodelabels:
        if data['node2labels'] is not None:
            nodelabels = data['node2labels']
        else:
            nodelabels = True
    else:
        nodelabels = None
    nodecolors = data['node2colors']

    fig, axs = plt.subplots(nrows=2, ncols=n_plots + 1, figsize=(n_plots * 4, 8))

    for i, kind in enumerate(['red', 'syn']):
        decomposed_edges = data['edges'][kind]
        all_edges = [edge for n in data['orders'] for edge in decomposed_edges[n]]
        ax = axs[i, 0]
        hyperplot.polygons(all_edges, ax=ax, nodecolors=nodecolors, nodelabels=nodelabels, internode_dist=internode_dists[i], **kwargs)
        ax.set_title(f"{kind.upper()}")

        for j, n in enumerate(decomposed_edges.keys()):
            ax = axs[i, j + 1]
            edges = decomposed_edges[n]
            if len(edges) > 0:
                hyperplot.polygons(edges, ax=ax, nodecolors=nodecolors, nodelabels=nodelabels, internode_dist=internode_dists[i], **kwargs)
            else:
                ax.axis('off')
            ax.set_title(f"Multiplet Order: {n}")

def plot_two_rows(data, column_spacing=2.5, nodesize=0.11, subplot_width=20, subplot_height=4):
    '''
    Plot O-info hypergraph using bipartite two row visualization from hypernetx

    Parameters
    ----------
    data : dict
        'edges' : {'red': {order (int) : list of edges}, 'syn': {order (int) : list of edges}}
        'node2labels' : {node : label}
        'node2colors' : {node : color}
        'orders' : list of int with multiplex orders (e.g. [3, 4, 5, 6])
        'nodeorder', list with order to plot nodes
    '''
    n_plots = len(data['orders'])
    nodelabels = data['node2labels']
    nodeorder = data['nodeorder']
    nodecolors = data['node2colors']

    fig, axs = plt.subplots(n_plots, 2, figsize=(subplot_width, n_plots * subplot_height))

    for i, kind in enumerate(['red', 'syn']):
        for n, order in enumerate(data['edges'][kind].keys()):
            ax = axs[i] if n_plots == 1 else axs[n, i]
            edges = data['edges'][kind][order]
            # hyperplot.two_rows(edges,
            #                    nodelabels=nodelabels,
            #                    nodecolors=nodecolors,
            #                    nodeorder=nodeorder,
            #                    ax=ax,
            #                    nodesize=nodesize,
            #                    column_spacing=column_spacing)
            hyperplot.ranktwo_rows(edges,20,histograms["synappear"], nodelabels=nodelabels,
                                                nodecolors=nodecolors,
                                                nodeorder=nodeorder,
                                                ax=ax,
                                                nodesize=nodesize,
                                                column_spacing=column_spacing)
            if n==1:
                ax.set_title(f'{kind.upper()}\nMultiplet Order: {order}', fontsize=16)
            else:
                ax.set_title(f'Multiplet Order: {order}', fontsize=16)

def plot_areas(data, edgecolors='gray'):
    '''
    Plot O-info hypergraph using concentric traces from hypernetx.

    Parameters
    ----------
    data : dict
        'edges' : {'red': {order (int) : list of edges}, 'syn': {order (int) : list of edges}}
        'node2labels' : {node : label}
        'node2colors' : {node : color}
        'orders' : list of int with multiplex orders (e.g. [3, 4, 5, 6])
    '''
    nodelabels = data['node2labels']
    nodecolors = data['node2colors']

    n_plots = len(data['orders'])
    fig, axs = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))
    
    for i, kind in enumerate(['red', 'syn']):
        for n, order in enumerate(data['edges'][kind].keys()):

            ax = axs[i] if n_plots == 1 else axs[i, n]
            edges = data['edges'][kind][order]

            if len(edges) > 0:
               
               
                hyperplot.areas(edges,
                               nodelabels=nodelabels,
                              nodecolors=nodecolors,
                             edgecolors=edgecolors,
                               ax=ax,


                             linewidth=1)
            else:
                ax.axis('off')

            ax.set_title(f'Multiplet Order: {order}')

def plot_planar(data):
    '''
    Plot planar hypergraph using networkx.

    Parameters
    ----------
    data : dict
        'edges' : {'red': {order (int) : list of edges}, 'syn': {order (int) : list of edges}}
        'node2labels' : {node : label}
        'node2colors' : {node : color}
        'orders' : list of int with multiplex orders (e.g. [3, 4, 5, 6])
    '''

    nodelabels = data['node2labels']

    n_plots = len(data['orders'])
    fig, axs = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))

    for _, ax in np.ndenumerate(axs):
        ax.axis('off')

    for i, kind in enumerate(['red', 'syn']):
        for n, order in enumerate(data['edges'][kind].keys()):
            ax = axs[i] if n_plots == 1 else axs[i, n]
            edges = data['edges'][kind][order]

            hyperplot.planar(edges, nodelabels=nodelabels, ax=ax)

            ax.set_title(f'Multiplet Order: {n + 3}')
            

   
if __name__ == "__main__":

    DATASET_DIR = Path.cwd() / 'data'
    SAVE_DIR = Path.cwd() / 'figs'

    savefig = True
    datasets = [ 'empathy','eating' ]
    #plots = ['polygons','planar','two_rows','areas']
   # datasets = ['empathy']
    plots = ['']
    for dataset in datasets:

        # LOAD DATA
        print(f'DATASET: {dataset.upper()}')
        print(8 * '=')
 
        if dataset=='eating':
             fpath = DATASET_DIR / 'EatingDisorders.mat'
             data = load_eating_dataset(fpath)
              
        elif dataset=='empathy':
            fpath = DATASET_DIR / 'Briganti2017.mat'
            data = load_empathy_dataset(fpath)

        else:
            raise ValueError('Dataset not accepted.')
        ##fig, axs = plt.subplots()
       
        # PLOT DATA
        #histograms=values_histogram(data)
        
        if 'two_rows' in plots:
            plot_two_rows(data, column_spacing=50, nodesize=0.6, subplot_width=25, subplot_height=10)
            plt.suptitle(f'{dataset.upper()} Dataset', fontsize=20)
            plt.subplots_adjust(top=0.9)
            if savefig:
                plt.savefig(SAVE_DIR / f"{dataset}_two-rows.png", dpi=300)
        
        if 'areas' in plots:
            plot_areas(data)
            plt.suptitle(f"{dataset.upper()} Dataset", fontsize=20)
            if savefig:
                plt.savefig(SAVE_DIR / f"{dataset}_areas.png", dpi=300)

        if 'planar' in plots:
            plot_planar(data)
            plt.suptitle(f'{dataset.upper()} Dataset', fontsize=20)
            if savefig:
               plt.savefig(SAVE_DIR / f"{dataset}_planar.png", dpi=300)
        
        if 'polygons' in plots:
                  plot_polygons(data, internode_dists=[1.6, None], show_nodelabels=True, **{'node_size':0.5})
                  plt.suptitle(f'{dataset.upper()} Dataset', fontsize=20)
                  if savefig:
                      plt.savefig(SAVE_DIR / f"{dataset}_polygons.png", dpi=300)
             