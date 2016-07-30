# -*- coding: utf-8 -*-
"""plotting.py: Visualization of networks, weights, and activations


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

plot_timeseries(): Plot different timeseries in a list vs an optional list 
    of target timeseries
timeseries_prediction_vs_target(): Plot different timeseries in list vs list
     of target timeseries (simpler alternative to plot_timeseries())
tc_layer_visualization(): Visualize maxlayer of meanlayer
timeseries_vs_preds_at_timesteps(): plot continuous timeseries vs 
    predictions at certain timesteps (e.g. lstm activations) with optional 
    markers and alpha
draw_lstm_unit(): Draw LSTM units in a LSTM network
net_visualization(): Visualize LSTM or other network with node activations
    and weights of connections.
weight_heatmap(): Create a heatmap plot for given weights

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-30  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')

from cycler import cycler

import matplotlib as mpl
from matplotlib import pyplot as pl
from matplotlib import patches as patches
pl.ioff()


def plot_timeseries(kwargs):
    """
    Plot different timeseries in list vs list optional of target timeseries
    """
    timeseries = kwargs.get('timeseries') #list(array([timesteps, 3])) contains min, max, mean
    filename = kwargs.get('filename')
    
    targets = kwargs.get('targets', [])
    ts_labels = kwargs.get('ts_labels', None)
    suptitle = kwargs.get('suptitle', '')
    title = kwargs.get('title', '')
    alpha = kwargs.get('alpha', 1.)
    balpha = kwargs.get('balpha', alpha/4)
    linewidth = kwargs.get('linewidth', 1.)
    linestyle_lines = kwargs.get('linestyle_lines', ['-', ':'])
    markes_lines = kwargs.get('markes_lines', ['', ''])
        
    fig, ax = pl.subplots(figsize=kwargs.get('figsize', [8,5]))
    
    
    if not isinstance(timeseries, list):
        timeseries = [timeseries]
    
    colors = [pl.cm.brg(i) for i in np.linspace(0, 0.9, len(timeseries))]
    for t_i, ts in enumerate(timeseries):
        ax.fill_between(np.arange(len(ts[:,1])), ts[:,1], ts[:,0], facecolor=colors[t_i], alpha=balpha, linestyle=':')
        ax.plot(ts[:,2], 
                linestyle=linestyle_lines[ts_labels[t_i].endswith('(train)')], 
                alpha=alpha, label=str(ts_labels[t_i]), linewidth=linewidth, 
                color=colors[t_i-ts_labels[t_i].endswith('(train)')],
                marker=markes_lines[ts_labels[t_i].endswith('(train)')])
    
    color = kwargs.get('target_color', 'r')
    for (t_i, tar) in enumerate(targets):
        line_styles = '-'
        ax.plot(tar, linestyle=line_styles, color=color, alpha=alpha, label='target', linewidth=0.6)
        
    #pl.axvline(x=118, linestyle='-.', color='r')
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    if kwargs.get('ylim', None) is not None:
        pl.ylim(kwargs.get('ylim'))
    pl.grid()
    pl.axis('on')
    pl.suptitle(suptitle)
    pl.title(title)#, fontsize=kwargs.get('fontsize', 10))
    
    pl.autoscale(enable=True)
    pl.tight_layout()
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax.legend(loc='center left', prop=kwargs.get('legend_props', {'size':10}), bbox_to_anchor=(1, 0.5))
    # make lines wider
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    
    pl.savefig(filename)
    return 0

def timeseries_prediction_vs_target(kwargs):
    """
    Plot different timeseries in list vs list of target timeseries
    """
    predictions = kwargs.get('predictions') #2dims (timesteps, predictions)
    targets = kwargs.get('targets')
    filename = kwargs.get('filename')
    
    suptitle = kwargs.get('suptitle', '')
    title = kwargs.get('title', '')
    alpha = kwargs.get('alpha', 1.)
        
    fig, ax = pl.subplots(figsize=[20,20])
    
    linestyle_lines = '-'
    
    
    for (n_i, n) in enumerate(np.arange(predictions.shape[1])):
        if n_i == 0:
            colors = [pl.cm.brg(i) for i in np.linspace(0, 0.9, predictions.shape[1]-1)]
            #ax.set_color_cycle(colors) #depreciated with matplotlib 1.5
            ax.set_prop_cycle(cycler('color', colors))
        else:
            ax.plot(predictions[:,n], linestyle=linestyle_lines, alpha=alpha, 
                    label=str(n), linewidth=0.4)
    
    ax.plot(predictions[:,0], linestyle='-', alpha=alpha, 
            label='prediction', color='k', linewidth=0.6)
    
    color='r'
    for (t_i, tar) in enumerate(targets):
        line_styles = '-'
        ax.plot(tar, linestyle=line_styles, color=color, alpha=alpha, label='target', linewidth=0.6)
        color='b'
        
    pl.axvline(x=118, linestyle='-.', color='r')
    ax.set_xlabel('timestep')
    ax.set_ylabel('activations')
    
    pl.grid()
    pl.axis('on')
    pl.suptitle(suptitle)
    pl.title(title, fontsize=10)
    pl.autoscale(enable=True)
    pl.savefig(filename)
    return 0

def tc_layer_visualization(kwargs):
    """
    Visualize maxlayer of meanlayer
    """
    y_axes = kwargs.get('y_axes')
    x_axis = kwargs.get('x_axis')
    activations = kwargs.get('activations')
    filename = kwargs.get('filename')
    
    suptitle = kwargs.get('suptitle', '')
    title = kwargs.get('title', '')
    coding_snps = kwargs.get('coding_snps', dict())
    alpha = kwargs.get('alpha', True)
    
    fig, ax = pl.subplots()
    
    linestyle_lines = '-'
    linestyle_coding = ':'
    marker_lines = '+'
    marker_maxact = 'o'
    size_maxact = 30
    colors = [pl.cm.brg(i) for i in np.linspace(0, 0.9, y_axes.shape[1])]
    
    #ax.set_color_cycle(colors)
    ax.set_prop_cycle(cycler('color', colors))
    
    activations = np.abs(activations / (np.max(np.abs(activations)) or 1.))
    activations = np.maximum(activations, 0.1)
    max_act_ind = np.argmax(y_axes, axis=0)
    for n in np.arange(y_axes.shape[1]):
        if alpha:
            base_line, = ax.plot(x_axis, y_axes[:,n], 
                                 linestyle=linestyle_lines, 
                                 marker=marker_lines, alpha=activations[n], 
                                 label=str(n))
            
            ax.scatter(x_axis[max_act_ind[n]], y_axes[max_act_ind[n],n], 
                    marker=marker_maxact, s=size_maxact, linewidths=0.1,
                    c=base_line.get_color(), alpha=activations[n], label="")
        else:
            base_line, = ax.plot(x_axis, y_axes[:,n], 
                                 linestyle=linestyle_lines, 
                                 marker=marker_lines, alpha=0.5, label=str(n))
            
            ax.scatter(x_axis[max_act_ind[n]], y_axes[max_act_ind[n],n], 
                    marker=marker_maxact, s=size_maxact, linewidths=0.1,
                    c=base_line.get_color(), alpha=0.5, label="")
    
    ax.set_xlabel('position')
    ax.set_ylabel('solid: tc_layer output')
    
    
    for key in list(coding_snps.keys()):
        if key == 'p':
            color = 'b'
        elif key =='a':
            color = 'r'
        else:
            color='k'
        
        for c, coding_snp in enumerate(coding_snps[key]):
            pl.axvline(x=coding_snp, linestyle=linestyle_coding, color=color, label=key if c == 0 else "")
    
    pl.axis('on')
    pl.suptitle(suptitle)
    pl.title(title, fontsize=10)
    pl.legend(prop={'size':6})
    pl.savefig(filename)
    return 0

def timeseries_vs_preds_at_timesteps(kwargs):
    """
    Plot continuous timeseries vs predictions at certain timesteps (e.g. 
    lstm activations) with optional markers and alpha
    """
    y_axes = kwargs.get('predictions') #predictions shape=(timesteps, outputs)
    x_axis = kwargs.get('timesteps') #timesteps to be plotted against
    
    impacts = kwargs.get('impacts', None) #optional impacts the predictions have (e.g. pred*sum(w_out)), can by positive and negative, lead to alpha
    alpha_max = kwargs.get('alpha_max', 0.7) #maximum alpha value for plots
    markers = kwargs.get('markers', list()) #optional vertical markers at given timesteps as [[x_pos, color], [x_pos, color], ...]
    suptitle = kwargs.get('suptitle', '')
    figsize = kwargs.get('figsize', (10,10))
    xlabel = kwargs.get('xlabel', 'position')
    ylabel = kwargs.get('ylabel', 'activation')
    savename = kwargs.get('savename', None)
    scale_x = kwargs.get('scale_x', 1)
    legend_size = kwargs.get('legend_size', 7)
    xlim = kwargs.get('xlim', dict())
    legend =kwargs.get('legend', True)
    
    fig, ax = pl.subplots(figsize=figsize)
    
    # line and marker styles
    prediction_lines = '-'
    prediction_markers = '+'
    markers_marker = ':'
    maxact_marker = 'o'
    maxact_markersize = 30
    
    # create and set colorcycle
    colors = [pl.cm.brg(i) for i in np.linspace(0, 0.9, y_axes.shape[1])]
    #ax.set_color_cycle(colors) #depreciated since matplotlib 1.5
    ax.set_prop_cycle(cycler('color', colors))
    
    # set alpha values according to impacts or alpha_max
    
    if impacts is not None:
        # normalize impacts to alphas
        impacts = np.abs(impacts / (np.max(np.abs(impacts)) or 1.)) * alpha_max
        # apply a minimum alpha
        alphas = np.maximum(impacts, 0.1)
        # find maximum activation in the predicted timeseries
        max_act_ind = np.argmax(y_axes, axis=0)
    else:
        alphas = np.ones(y_axes.shape[1]) * alpha_max
        max_act_ind = np.argmax(y_axes, axis=0)
    
    
    # plot predictions
    for n in np.arange(y_axes.shape[1]):
        base_line, = ax.plot(x_axis*scale_x, y_axes[:,n], 
                             linestyle=prediction_lines, 
                             marker=prediction_markers, alpha=alphas[n], 
                             label=str(n))
        
        ax.scatter(x_axis[max_act_ind[n]]*scale_x, y_axes[max_act_ind[n],n], 
                marker=maxact_marker, s=maxact_markersize, linewidths=0.1,
                c=base_line.get_color(), alpha=alphas[n], label="")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # plot markers
    if isinstance(markers,dict):
        for key in markers:
            for m in markers[key]:
                pl.axvline(x=m*scale_x, linestyle=markers_marker, color=key)
    elif isinstance(markers,list):
        for m in markers:
            pl.axvline(x=m[0]*scale_x, linestyle=markers_marker, color=m[1])
    
    ax.axis('on')
    ax.grid()
    pl.title(suptitle)
    pl.tight_layout()
    pl.autoscale(True)
    ax.set_xlim(**xlim)
    #h0, l0 = ax.get_legend_handles_labels()
    # Shrink current axis by 20%
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', prop={'size':legend_size}, bbox_to_anchor=(1, 0.5))
    
    
    pl.savefig(savename)
    return 0

def draw_lstm_unit(ax, scaling=(1.,1.), position=(0.,0.), node_radius=None, 
                   node_colors=['b','b','b','b','b']):
    """
    Draw LSTM units in a LSTM network
    """
    #colors.colorConverter.to_rgba('b', alpha=0.1)
    # list to store all created patches
    
    node_tags = ['input', 'input gate', 'forget gate', 'output gate', 'output']
    dt_nodes = [('x_pos', '<f4'), ('y_pos', '<f4')]
    node_pos = np.zeros((5), dtype=dt_nodes) # [in, ingate, forget, outgate, out] as [x_pos,y_pos]
    
    lstm_width = 0.9 * scaling[0]
    
    if node_radius == None:
        node_radius = lstm_width / 4. * 1.5  / 2.
        lstm_size = np.array([lstm_width, node_radius*5.])
    else:
        lstm_size = np.array([lstm_width, max(lstm_width / 8.  / 2., 
                                              node_radius)*4.])
        node_radius = min(lstm_width / 8.  / 2., node_radius)
    
    
    
    # center lstm units
    position = tuple(np.subtract(position, (lstm_width / 2., 0)))
    
    node_pos[:-1]['x_pos'] = np.linspace(position[0]+node_radius*1.5,
                                    position[0]+lstm_size[0]-node_radius*1.5,
                                    4)
                                    
    node_pos[:-1]['y_pos'] = position[1]
    node_pos[-1] = tuple(position + lstm_size * [0.5,1.]) 
    ax.add_patch(patches.Rectangle(position, width=lstm_size[0],
                                   height=lstm_size[1], alpha=0.1,
                                   facecolor="y", zorder=2))
    for n_i, n in enumerate(node_pos):
        ax.add_patch(patches.Circle(n,node_radius,fc=node_colors[n_i], ec='none', zorder=2))
        if n_i < 4:
            ax.text(n['x_pos'], n['y_pos']+node_radius, node_tags[n_i], 
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=20.*scaling[0], zorder=3)
        else:
            ax.text(n['x_pos'], n['y_pos']-node_radius, node_tags[n_i], 
                    horizontalalignment='center', verticalalignment='top',
                    fontsize=20.*scaling[0], zorder=3)
    
    return node_pos

def net_visualization(kwargs):
    """ Visualize network with node activations and weights of connections.
    
    Parameters
    ----------
    n_nodes : 1D numpy array
        number of neurons per layer
    weights : list of 2D numpy arrays
        each list element represents the weights between two layers and consist 
        of a numpy array of shape (n_nodes_current_layer * n_nodes_next_layer),
        resulting in a list lenght of n_layers-1.
        For LSTM units, a list element holds a list with numpy arrays of shape 
        (n_nodes_current_layer * n_nodes_next_layer) describing the 
        foward connections for ['input', 'input gate', 'forget gate', 
        'output gate']. 
    activations : list of 1D numpy arrays
        each list element represents the activations in a layer and consist
        of a numpy array of shape (n_nodes_current_layer),
        resulting in a list lenght of n_layers
    top_layer : lasagne layer
        top (last) layer of network, lower layers not covered by n_nodes are 
        ignored
        
    Returns
    -------
    plots figure
    
    Notes
    -----
    TODO: recurrent LSTM connections
    """
    
    n_nodes = kwargs.get('n_nodes')
    weights = kwargs.get('weights')
    activations = kwargs.get('activations')
    filename = kwargs.get('filename', 400.)
    
    suptitle = kwargs.get('suptitle', '')
    title = kwargs.get('title', '')
    nodewidth = kwargs.get('nodewidth', 0.8) # relative to data
    linewidth = kwargs.get('linewidth', 30.) # relative to view
    canvas_range = kwargs.get('canvas_range', np.array([0.,1.,0.,1.]))
    figure_size = kwargs.get('figure_size', np.array([10.,10.]))
    inputs = kwargs.get('inputs', None)
    activation_function = kwargs.get('activation_function', 'linear')
    
    net_nodes = list()
    net_weights = list(weights)
    dt_nodes = [('act', '<f4'), ('x_pos', '<f4'), ('y_pos', '<f4')]
    
    
    # Create figure
    fig = pl.figure(figsize=figure_size)
    
    
    # Calculate colorbar plot specs (values relative to [0, 1, 0, 1])
    cbar_width = 0.05
    cbar_dist = 0.02
    cbar_space = cbar_width + cbar_dist
    
    cmap = pl.get_cmap('RdYlGn')
    cbar_vsep = 0.1
    cbar_height = (1.-(cbar_space*2.)) / (len(net_weights)) - cbar_vsep
    cbar_vpos = np.arange(0.+cbar_space+cbar_vsep/2, 1.-(cbar_space*2.), cbar_height+cbar_vsep)
    
    # Calculate network plot specs
    smallest_ax_range = min(canvas_range[1::2] - canvas_range[::2])
    node_radius = smallest_ax_range * nodewidth / 2. / max([max(n_nodes), 
                                                           len(n_nodes)])
    
    
    linewidth /= max([max(n_nodes), len(n_nodes)])
    
    # Create axis for network plot
    net_plot_pos = [0., 0.+cbar_space, 1.-(cbar_space*2.), 1.-(cbar_space*2.)] #[left, bottom, width, height]
    ax = fig.add_axes(net_plot_pos)
    
    ax.axis(canvas_range + [-node_radius*3., node_radius*3., 
                            -node_radius*3., node_radius*3.])
    ax.set_frame_on(False)
    ax.set_autoscale_on(False)
    ax.set_aspect(aspect='equal',adjustable='box')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    
    
    # Calculate vertical position of evenly spaced layers
    v_spacing = np.linspace(canvas_range[2], canvas_range[3], len(n_nodes))
    
    # Calculate horizontal position of evenly spaced neurons per layers
    for l in np.arange(len(n_nodes)-1):
        net_nodes.append(np.zeros(n_nodes[l], dtype=dt_nodes))
        if n_nodes[l] == 1:
            net_nodes[-1]['x_pos'] = canvas_range[0] + \
                                     (canvas_range[1] - canvas_range[0]) / 2.
        else:
            net_nodes[-1]['x_pos'] = np.linspace(canvas_range[0], 
                                                 canvas_range[1], n_nodes[l])
        net_nodes[-1]['y_pos'] = v_spacing[l]
        net_nodes[-1]['act'] = activations[l]
    else:
        l += 1
        net_nodes.append(np.zeros(n_nodes[l], dtype=dt_nodes))
        if n_nodes[l] == 1:
            net_nodes[-1]['x_pos'] = canvas_range[0] + \
                                     (canvas_range[1] - canvas_range[0]) / 2.
        else:
            net_nodes[-1]['x_pos'] = np.linspace(canvas_range[0], 
                                                 canvas_range[1], n_nodes[l])
        net_nodes[-1]['y_pos'] = v_spacing[l]
        net_nodes[-1]['act'] = activations[l]
    
    # Plot connections (=weights)
    for layer, layer_weights in enumerate(net_weights):
        
        if isinstance(layer_weights, list):
            # lstm layer
            max_weight = max([np.max(np.abs(node_weights)) \
                                  for node_weights in layer_weights]) or 1.
            
            nr_lstms = layer_weights[0].shape[1]
            lstm_scaling = (canvas_range[1] - canvas_range[0]) / nr_lstms
            dt_lstm_nodes = [('x_pos', '<f4'), ('y_pos', '<f4')]
            lstm_pos = np.zeros((5, nr_lstms), dtype=dt_lstm_nodes)
            
            lstm_x_pos = np.linspace(canvas_range[0] + lstm_scaling / 2., 
                                     canvas_range[1] - lstm_scaling / 2.,
                                     nr_lstms).astype('<f4')
            
            for lstm_unit, lstm_spec in enumerate(net_nodes[layer+1]):
                lstm_pos[:, lstm_unit] = \
                    draw_lstm_unit(ax, scaling=(lstm_scaling,lstm_scaling),
                               position=(lstm_x_pos[lstm_unit],
                                         lstm_spec['y_pos']),
                               node_radius=node_radius,
                               node_colors=['y','y','y','y','y'])
            for lstm_n, lstm_node in enumerate(lstm_pos[:-1]):
                for n1  in np.arange(layer_weights[lstm_n].shape[0]):
                    for n2 in np.arange(layer_weights[lstm_n].shape[1]):
                        weight_color = (layer_weights[lstm_n][n1,n2] / 
                                                max_weight + 1) / 2.
                        weight_alpha = abs(layer_weights[lstm_n][n1,n2] / 
                                                max_weight)
                        ax.plot([net_nodes[layer]['x_pos'][n1],
                                 lstm_node[n2]['x_pos']],
                                [net_nodes[layer]['y_pos'][n1],
                                 lstm_node[n2]['y_pos']], 
                                linestyle='-', linewidth=linewidth, 
                                alpha=weight_alpha, 
                                color=cmap(weight_color), zorder=1)
            
            # prepare connections to next layer
            net_nodes[layer+1]['x_pos'] = lstm_pos[-1,:]['x_pos']
            net_nodes[layer+1]['y_pos'] = lstm_pos[-1,:]['y_pos']
        else:
            # conventional layer
            max_weight = np.max(np.abs(layer_weights)) or 1.
            for n1  in np.arange(layer_weights.shape[0]):
                for n2 in np.arange(layer_weights.shape[1]):
                    weight_color = (layer_weights[n1,n2] / 
                                            max_weight + 1) / 2.
                    weight_alpha = abs(layer_weights[n1,n2] / 
                                            max_weight)
                    ax.plot([net_nodes[layer]['x_pos'][n1],
                             net_nodes[layer+1]['x_pos'][n2]],
                            [net_nodes[layer]['y_pos'][n1],
                             net_nodes[layer+1]['y_pos'][n2]], 
                            linestyle='-', linewidth=linewidth, 
                            alpha=weight_alpha, 
                            color=cmap(weight_color), zorder=1)
        
        cbax = fig.add_axes([1.-(cbar_space*2.), cbar_vpos[layer], cbar_width, cbar_height]) #[left, bottom, width, height]
        norm = mpl.colors.Normalize(vmin=-max_weight, vmax=max_weight)
        cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,
                                       norm=norm,
                                       orientation='vertical')
        
        cb.set_ticks(np.linspace(-max_weight,max_weight,5.))
        cb.ax.set_autoscale_on('True')
        
    
    if inputs == None:
        for nodes in net_nodes:
            if activation_function != 'sigmoid' and len(nodes)>1:
                scaled_acts = nodes['act'] / (np.nanmax(np.abs(nodes['act'])) or 1.)
            else:
                scaled_acts = nodes['act']
            for node_n, node in enumerate(nodes):
                scaled_act = scaled_acts[node_n]
                ax.add_patch(patches.Circle((node['x_pos'], node['y_pos']),
                                            node_radius,
                                            fc=cmap((scaled_act+1.)/2.),
                                            ec='none', zorder=2))
                
    else:
        for n, nodes in enumerate(net_nodes):
            if n == 0:
                ax_size_change = 0
                for k in list(inputs.keys()):
                    for i_n, input_layer in enumerate(inputs[k]):
                        add_v_space = ax_size_change + i_n*node_radius*1.5
                        for node_n, node in enumerate(nodes):
                            if k == 'p':
                                pts = np.array([[node['x_pos']-node_radius/2., 
                                                 node['y_pos']], 
                                                [node['x_pos']+node_radius/2., 
                                                 node['y_pos']],
                                                [node['x_pos'], 
                                                 node['y_pos']+node_radius]])
                            else:
                                pts = np.array([[node['x_pos']-node_radius/2., 
                                                 node['y_pos']+node_radius], 
                                                [node['x_pos']+node_radius/2., 
                                                 node['y_pos']+node_radius],
                                                [node['x_pos'], 
                                                 node['y_pos']]])
                            input_nodes = input_layer[node_n]
                            pts -= [0,add_v_space+node_radius/2.]
                            ax.add_patch(patches.Polygon(pts, closed=True,
                                            fc=cmap((input_nodes+1.)/2.),
                                            ec='none', zorder=2))
                            
                    ax_size_change = add_v_space + node_radius*1.5
                ax_size_change -= i_n*node_radius*1.5
                
                ax.set_ylim((ax.get_ylim()[0]-ax_size_change, ax.get_ylim()[1]))
            else:
                
                if activation_function != 'sigmoid' and len(nodes)>1:
                    scaled_acts =  nodes['act']/ (np.nanmax(np.abs(nodes['act'])) or 1.)
                else:
                    scaled_acts = nodes['act']
                
                for node_n, node in enumerate(nodes):
                    scaled_act = scaled_acts[node_n]
                    ax.add_patch(patches.Circle((node['x_pos'], node['y_pos']),
                                            node_radius,
                                            fc=cmap((scaled_act+1.)/2.),
                                            ec='none', zorder=2))
                                            
    fig.suptitle(suptitle + '\n' + title)
    pl.savefig(filename)
    
    return 0


def weight_heatmap(kwargs):
    """
    Create a heatmap plot for given weights, wrap plot at max_maplength
    """
    x = kwargs.get('x')
    max_maplength = kwargs.get('max_maplength')
    title = kwargs.get('title')
    
    add_ylines = kwargs.get('add_ylines', None)
    figsize = kwargs.get('figsize', (20,20))
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    savename = kwargs.get('savename', None)
    plot = kwargs.get('plot', False)
    
    # convert 1dimensional to 2dimensional matrix
    if x.ndim < 2:
        x = x[None,:]
    
    # wrap plot to desired length
    divider = np.ceil(float(x.shape[0]) / max_maplength)
    
    # padd axis to length
    pad_len = np.int(max_maplength * divider - x.shape[0])
    if pad_len:
        x = np.pad(x, pad_width=((0,pad_len),(0,0)), 
                   mode='constant', constant_values=(0.,))
    
    # create figure, account for colormap size
    fig, axes = pl.subplots(ncols=int(divider)+1, 
                            gridspec_kw=dict(width_ratios=np.append(\
                                                np.ones(divider), [0.5])), 
                            figsize=figsize)
    
    # create colormap
    x_abs = np.abs(x)
    max_x = np.max(x_abs)
    norm = matplotlib.colors.Normalize(vmin=-max_x, vmax=max_x)
    cb = matplotlib.colorbar.ColorbarBase(axes[-1], 
                                          cmap=matplotlib.cm.coolwarm,
                                          norm=norm, orientation='vertical')
    #cb.set_ticks(np.linspace(-max_x,max_x,5.))
    cb.ax.set_autoscale_on('True')
    
    # plot heatmaps (one subset per column)
    div_shape = x.shape[0] / divider
    # use higher ytick frequency
    ticklocator = mpl.ticker.MultipleLocator(base=5.0)
    for i in np.arange(divider, dtype=int):
        axes[i].imshow(x[i*div_shape:(i+1)*div_shape,:],
                       cmap=matplotlib.cm.coolwarm, aspect='auto', 
                       interpolation='nearest', 
                       extent=[0, x.shape[1], (i+1)*div_shape, i*div_shape],
                       vmin=-max_x, vmax=max_x)
        axes[i].set_xticks(np.arange(x.shape[1]))
        # use higher ytick frequency
        axes[i].yaxis.set_major_locator(ticklocator)
        # add y-axis label on very left heatmap
        if i == 0:
            if ylabel != None:
                axes[i].set_ylabel('{}'.format(ylabel))
        # add x-axis labels on every heatmap
        if xlabel != None:
            axes[i].set_xlabel('{}'.format(xlabel))
    
    
    # add plot title and make some final scaling adjustments
    pl.suptitle(title, fontweight='bold')  
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4, top=0.95)
    
    for axis in axes:
        axis.set_aspect('auto')
        axis.set_autoscale_on(True)
    
    if add_ylines != None:
        for add_yline in add_ylines:
            ax_i = int(add_yline[0]['y'] / div_shape)
            axes[ax_i].axhline(**add_yline[0])
            axes[ax_i].text(*add_yline[1])
    
    # save figure
    if savename != None:
        pl.savefig(savename)
    
    # plot figure
    if plot:
        fig.show()
        return fig, axes #could return figure alternatively
    else:
        pl.close()
        return 0 #could return figure alternatively
    
