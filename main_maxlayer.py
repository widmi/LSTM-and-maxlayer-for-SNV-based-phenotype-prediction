# -*- coding: utf-8 -*-
"""main_maxlayer.py: Core file for maxlayer experiments


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Call this file with maxlayer_start_nested_cross-validation.py or 
maxlayer_start_cross-validation.py.
build_model(): Build maxlayer network
costfun(): Function for the calculation of the loss/error/cost
create_iter_functions(): Create theano functions for weight updates etc.
maxlayer_visualization(): Plot maxlayer activations or network weights
train(): Create network training loop
main(): Perform the experiment by building, training, and saving the network
start_experiment(): Call main() to start an experiment (to be called as
                    sub-process)

Note: Structure and some parts taken from lasagne tutorial

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-28  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""


############################################################################
#  Imports                                                                 #
############################################################################


import sys
from os import path

# theano is buggy when multithreading -> start plotting deamon before theano
from michaels_modules.utility.plotting_daemons import start_plotting_daemon
(plotting_queue, plotting_proc) = start_plotting_daemon()

# import garbage collector for fast removal of minibatches
import gc

import theano
import theano.tensor as T

import lasagne
from lasagne import init
from lasagne.layers import get_output

import time
import numpy as np
import itertools
from collections import OrderedDict
import re
import gzip
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as pl
pl.ioff()

# Custom modules
from michaels_modules.utility import plotting as mypl
from michaels_modules.utility.external_sources import Tee, ConfusionMatrix
from michaels_modules.net_utils.save_load import savemodel, loadmodel

from michaels_modules.net_utils.nonlinearities import rectify, categorical_crossentropy_logdomain, elu, erth
from michaels_modules.net_utils.regularization import constr_weight_sum

from michaels_modules.net_utils.layers import MaxLayer, MeanLayer
from michaels_modules.net_utils.input_encoders_numba import InputEncoderTriangles, InputEncoderBinary
from michaels_modules.net_utils.hdf5_mb_read import HDF5_Batch_Reader, minibatch_generator


############################################################################
#  GLOBALS                                                                 #
############################################################################

# Theano behavior with unused inputs to functions
on_unused_input='warn'

# In case of artificial datasets, the coding snps are determined from the
#  filename and stored in coding_snps for plotting
coding_snps=[]

# Datatype for saving plots
image_datatype = '.png'

# Plot and save image at every plot_ep epochs
plot_ep = 5000

# This it not threadsafe, will be corrected for in main()
np.random.seed(123)
rnd_gen = np.random.RandomState(seed=123)
lasagne.random.set_rng(rnd_gen)


############################################################################
#  Build Network                                                           #
############################################################################

def build_model(dataset, layerspecs, updatespecs=None):
    """
    Build the lasagne network given the specifications
    
    Parameters
    -------
    dataset : dict
        A dataset dictionary as created by HDF5_Batch_Reader(), containing
        necessary information about the dataset sizes and dimensions
    layerspecs : dict
        Specifications for the network design, such as layer sizes, dropout,
        etc.. See maxlayer_start.py for an example.
    updatespecs : dict
        Dictionary with the update specifications and regularization
        methods. Required here sinze the negative weights penalty nwp
        requires a different initialization of the weights.
    Returns
    -------
    layers : OrderedDict()
        Dictionary with lasagne layers. Output layer is
        layers['output_layer']
    Notes
    -------
    Since maxlayer or meanlayer can be built, these layertypes are referred
    to as tc (timestep-convolution) layers.
    """
    
    # Initialize dropout rates and noise to 0, if not specified
    layerspecs['tc_do'] = layerspecs.get('tc_do', False)
    layerspecs['dense_do'] = layerspecs.get('dense_do', False)
    layerspecs['input_do'] = layerspecs.get('input_do', False)
    layerspecs['inputnoise'] = layerspecs.get('inputnoise', False)
    
    # Check for negative weights penalty in layerspecs, then 
    #  updatespecs['regularization']
    if layerspecs.get('nwp', False):
        updatespecs['regularization']['nwp'] = layerspecs['nwp']
    else:
        updatespecs['regularization']['nwp'] =\
            updatespecs['regularization'].get('nwp',  False)
    
    # TC layer parameters
    if layerspecs['tc_layer'] > 0:
        
        # Change initialization of weights and bias in TC layer, depending
        #  on nwp and used activation functions
        
        if updatespecs['regularization']['nwp']:
            w_sum = 5. # desired sum of weights -> sigmoid(5) = 0.9933071490
            # bias
            b_tc = init.Constant(-1)
            if layerspecs['input_encoder'] == 'binary':
                w_mean = w_sum / (dataset['n_features'] / 2)
            else:
                w_mean = w_sum / (dataset['input_encoder'].n_binary_bits / 2
                                  + 1)
        else:
            # bias
            b_tc = init.Constant(0.)
        
        if layerspecs['act'] == 'elu':
            # weigths and bias
            if updatespecs['regularization']['nwp']:
                w_tc = init.Uniform(1e-6,2*w_mean) #mean ~ w_mean
            else:
                w_tc = init.HeNormal('relu')
            
            # activation function
            a_tc = elu()
            
        elif layerspecs['act'] == 'relu':
            # weigths
            if updatespecs['regularization']['nwp']:
                w_tc = init.Uniform(1e-6,2*w_mean) #mean ~ w_mean
            else:
                w_tc = init.HeNormal('relu')
            
            # activation function
            a_tc = rectify()
            
            
        elif layerspecs['act'] == 'erth':
            # weigths
            #w_tc = init.HeNormal('relu')
            if updatespecs['regularization']['nwp']:
                w_tc = init.Uniform(1e-6,2*w_mean) #mean ~ w_mean
            else:
                w_tc = init.HeNormal()
            
            # activation function
            a_tc = erth(alpha=1e-1)
            
        elif layerspecs['act'] == 'sigmoid':
            # weigths
            #w_tc = init.HeNormal('relu')
            if updatespecs['regularization']['nwp']:
                w_tc = init.Uniform(1e-6,2*w_mean) #mean ~ w_mean
            else:
                w_tc = init.HeNormal()
            
            # activation function
            a_tc = lasagne.nonlinearities.sigmoid
    
    
    # Dense layer parameters
    if len(layerspecs['dense']) > 0:
        
        b_dense = init.Constant(0.)
        
        if layerspecs['dact'] == 'identity':
            a_dense = lasagne.nonlinearities.identity
            w_dense = init.HeNormal(1.)
        elif layerspecs['dact'] == 'sigmoid':
            a_dense = lasagne.nonlinearities.sigmoid
            w_dense = init.HeNormal(1.)
        elif layerspecs['dact'] == 'relu':
            a_dense = rectify()
            w_dense = init.HeNormal('relu')
        elif layerspecs['dact'] == 'elu':
            a_dense = elu()
            w_dense = init.HeNormal('relu')
    
    
    # Output layer parameters
    a_out = lasagne.nonlinearities.sigmoid
    w_out = init.HeNormal(1.)
    b_out = init.Constant(0.)
    
    
    # Create layers and store them in a OrderedDict()
    layers = OrderedDict()
    
    layers['input_layer'] =\
        lasagne.layers.InputLayer(shape=dataset['batch_size_X'])
    layers['input_layer_mod'] = layers['input_layer']
    
    if layerspecs['inputnoise']:
        layers['input_layer_mod'] =\
            lasagne.layers.GaussianNoiseLayer(layers['input_layer_mod'], 
                                              sigma=3./3)
    if layerspecs['input_do'] > 0:
        layers['input_layer_mod'] =\
            lasagne.layers.DropoutLayer(layers['input_layer_mod'], 
                                        p=layerspecs['input_do'],
                                        rescale=True)
    
    if layerspecs['tc_layer'] > 0:
        
        # reshape input layer for input to TC layer/convolution
        layers['dense_premax_rs'] =\
            lasagne.layers.ReshapeLayer(layers['input_layer_mod'], 
                        (dataset['batch_size']*dataset['max_len'],
                         dataset['n_features']))
        
        # use a standard dense layer for convolutional part of the TC layer
        layers['dense_premax']=\
            lasagne.layers.DenseLayer(\
                layers['dense_premax_rs'], num_units=layerspecs['tc_layer'], 
                W=w_tc, nonlinearity=a_tc, b=b_tc)
        
        # reshape to prepare for maximum/mean of activations per sample
        layers['premax_rs'] =\
            lasagne.layers.ReshapeLayer(layers['dense_premax'], 
                        (dataset['batch_size'], dataset['max_len'], 
                         layerspecs['tc_layer']))
        
        # use custom layer to calculate maximum or mean of activations
        layerspecs['tc'] = layerspecs.get('tc', 'max') # default is maxlayer
        if layerspecs['tc'] == 'mean':
            layers['tc'] =\
                MeanLayer(layers['premax_rs'], axis=1)
        else:
            layers['tc'] =\
                MaxLayer(layers['premax_rs'], axis=1)
        
        # apply dropout on TC units
        if layerspecs['tc_do']:
            layers['tc_out'] = lasagne.layers.DropoutLayer(layers['tc'], 
                                        p=layerspecs['tc_do'], rescale=True)
        else:
            layers['tc_out']  = layers['tc']
        
        # set TC layer as input to dense layers
        layers['dense_in'] = layers['tc_out']
        
    else:
        # if no TC is used, forward the input directly to the dense layer
        layers['dense_in'] = layers['input_layer_mod']
    
    # Create dense layers
    if len(layerspecs['dense']) > 0:
        for d in range(len(layerspecs['dense'])):
            # create single dense layers and stack them
            if d == 0:
                layers['dense'+str(1+d)]=\
                    lasagne.layers.DenseLayer(\
                        layers['dense_in'],
                        num_units=layerspecs['dense'][d], 
                        W=w_dense, nonlinearity=a_dense, b=b_dense)
            else:
                layers['dense'+str(1+d)]=\
                    lasagne.layers.DenseLayer(\
                        layers['dense'+str(d)+'_out'],
                        num_units=layerspecs['dense'][d],
                        W=w_dense, nonlinearity=a_dense, b=b_dense)
            
            # applied dropout to dense units
            if layerspecs['dense_do']:
                layers['dense'+str(1+d)+'_out'] =\
                    lasagne.layers.DropoutLayer(layers['dense'+str(1+d)], 
                                                p=layerspecs['dense_do'],
                                                rescale=True)
            else:
                layers['dense'+str(1+d)+'_out'] = layers['dense'+str(1+d)]
        
        # set output of dense layer as input to output layer
        layers['pre_out'] = layers['dense'+
                                    str(len(layerspecs['dense']))+'_out']
        
    else:
        # otherwise forward directly to the output layer
        layers['pre_out'] = layers['dense_in']
    
    # Create output layer (dense layer with nr nodes = nr of classes)
    layers['output_layer'] = \
        lasagne.layers.DenseLayer(\
            layers['pre_out'], num_units=dataset['n_classes'], W=w_out, 
            nonlinearity=a_out, b=b_out)
    
    # Return the built network layers
    return layers


############################################################################
#  Cost function                                                           #
############################################################################

def costfun(prediction, target, function, mode, weights=None,
            weights_binarization=False, verbose=True):
    """
    Calculate costs from predictions and targets with lasagne/theano
    
    Parameters
    -------
    prediction : theano tensor
        predictions
    target : theano tensor
        target values
    function : string
        'squared_error', 'binary_crossentropy', or
        'categorical_crossentropy' for respective lasagne functions
    mode : string
        'normalized_sum', 'mean', or 'sum' for how the loss shall be
        aggregated over samples
    weights : theano tensor
        weights of individual sample errors
    weights_binarization : bool
        if True, the weights will be treated as binary; default: False
    Returns
    -------
    loss : theano function
        theano function for cost calculation
    """
    if verbose:
        print("\tUsed loss function: {}\n\nmode: {}".format(function, mode),
              end=' ')
        if weights == None:
            print("...")
        else:
            print(", weights binarization: {}...".format(\
                                                    weights_binarization))
    
    if weights_binarization and (weights is not None):
        weights = weights > 0
    
    if function == 'squared_error':
        loss = lasagne.objectives.squared_error(prediction, target)
    elif function == 'binary_crossentropy':
        loss = lasagne.objectives.binary_crossentropy(prediction, target)
    elif function == 'categorical_crossentropy':
        # lasagne calculation seems numerically instable -> use custom one
        loss = categorical_crossentropy_logdomain(prediction, target)
        #loss = lasagne.objectives.categorical_crossentropy(prediction, 
        #                                                   target)
    else:
        raise ValueError("Loss function not implemented!")
    
    loss = lasagne.objectives.aggregate(loss, weights=weights, mode=mode)
    return loss


############################################################################
#  Cost function                                                           #
############################################################################

def create_iter_functions(dataset, updatespecs, layers, layerspecs):
    """
    Create theano functions for weight updates, regularization, prediction,
    etc.
    
    Parameters
    -------
    dataset : dict
        A dataset dictionary as created by HDF5_Batch_Reader(), containing
        necessary information about the dataset sizes and dimensions
    updatespecs : dict
        Dictionary with the update specifications and regularization
        methods. ´
    layers : dict
        Dictionary with lasagne layers as created by build_model()
    Returns
    -------
    dict()
        Dictionary with functions for train: calculation of loss with weight
        update; cost: calculation of loss without weight update;
        cost_noreg: calculation of loss without weight update without
        penalties; pred: predictions without weight updates; shared: shared
        theano variables
    Notes
    -------
    Since maxlayer or meanlayer can be built, these layertypes are referred
    to as tc (timestep-convolution) layers.
    """

    # create symbolic variables
    sym_input = T.tensor3('input')   # float32
    sym_target = T.imatrix('target_output')  # integer symbolic variable
    sym_timesteps = T.ivector('timesteps')
    sym_cost_weights = T.fmatrix('cost_weights')
    
    # shared variables
    sh_input = theano.shared(np.zeros(shape=dataset['batch_size_X'], 
                    dtype=theano.config.floatX), borrow=True)
    
    sh_target = theano.shared(np.zeros(shape=(dataset['batch_size_y']),
                    dtype=theano.config.floatX), borrow=True)
    
    sh_timesteps = theano.shared(\
                        np.zeros(shape=dataset['batch_size_last_ind'], 
                                 dtype=np.int32),
                        borrow=True)
    
    sh_cost_weights = theano.shared(\
                        np.zeros(shape=dataset['batch_size_weights'],
                                 dtype=theano.config.floatX), 
                        borrow=True)
    
    # inputs for network
    input_dict = {layers['input_layer']: sym_input}
    
    
    # input node slices
    if layerspecs['input_encoder'] == 'triangles':
        # structure: features, ticker step, binary enc, triangle enc
        ticker_step_pos = (dataset['batch_size_X'][2] -
                            dataset['input_encoder'].n_inputnodes)
        positionnodes_pos = ticker_step_pos + 1
        triang_pos = (positionnodes_pos + 
                        dataset['input_encoder'].n_binary_bits)
        binary_pos = positionnodes_pos
        #snp_type_pos = 0
        
        #positions_slice = slice(positionnodes_pos, None) #without ticker steps
        triangle_slice = slice(triang_pos, None)
        binary_slice = slice(binary_pos, triang_pos)
        #snp_type_slice = slice(snp_type_pos, ticker_step_pos)
    elif layerspecs['input_encoder'] == 'binary':
        #snp_type_slice = slice(None, -(dataset['input_encoder'].n_inputnodes))
        binary_slice = slice(-(dataset['input_encoder'].data_nodes), None)
    
    print("CREATING FUNCTIONS...")
    print("\tCOSTS...")
    
    # Fetch update parameters
    function = updatespecs.get('loss_function')
    mode = updatespecs.get('mode', 'normalized_sum') # 'mean', 'sum'
    
    # Functions for output predictions
    prediction_train = get_output(layers['output_layer'], inputs=input_dict, 
                                  timesteps=sym_timesteps,
                                  deterministic=False)
    prediction = get_output(layers['output_layer'], inputs=input_dict, 
                            timesteps=sym_timesteps, deterministic=True)
    
    # Functions for cost calculations
    cost_train = costfun(prediction_train, sym_target, function, mode, 
                         weights=sym_cost_weights)
    
    cost = costfun(prediction, sym_target, function, mode=mode, 
                       weights=sym_cost_weights)
    
    cost_noreg = costfun(prediction, sym_target, function, mode=mode,
                       weights=sym_cost_weights)
    
    
    print("\t\t...DONE")
    
    print("\tREGULARIZATIONS...")
    
    # Fetch regularization parameters
    reg_fun_name = updatespecs['regularization'].get('function', None)
    
    if reg_fun_name != None:
        # Regularization weights for layers
        reg_weights = updatespecs['regularization']['weights']
        dense_layers = {layers['output_layer']: reg_weights[2]}
        for l in range(len(layerspecs['dense'])):
            dense_layers[layers['dense'+str(1+l)]] = reg_weights[1]
        ts_layers = dict()
        if layerspecs['tc_layer'] > 0:
            ts_layers[layers['dense_premax']] = reg_weights[0]
    
    # Functions for regularization calculation
    if reg_fun_name == None:
        pass
    elif reg_fun_name == 'l1':
        reg_fun_dense = lasagne.regularization.l1
        reg_fun_ts = lasagne.regularization.l1
    elif reg_fun_name == 'l2':
        reg_fun_dense = lasagne.regularization.l2
        reg_fun_ts = lasagne.regularization.l2
    elif reg_fun_name == 'reg_l2pergate':
        reg_fun_dense = lasagne.regularization.l2
        reg_fun_ts = lasagne.regularization.l2
    else:
        raise ValueError("Specified regularization function not implemented")
    
    if reg_fun_name != None:
        penalty_dense =\
                lasagne.regularization.regularize_layer_params_weighted(\
                    dense_layers, reg_fun_dense)
        penalty_ts = \
                lasagne.regularization.regularize_layer_params_weighted(\
                    ts_layers, reg_fun_ts)
        cost_train = cost_train + penalty_dense + penalty_ts
        cost = cost + penalty_dense + penalty_ts
    if layerspecs['tc_layer'] > 0:
        if updatespecs['regularization'].get('nwp', False):
            cost_train = (cost_train + 
                          T.sum(T.minimum(0, layers['dense_premax'].W)**2))
        if updatespecs['regularization'].get('l2_fanout', False):
            print("\t\tApplying l2_fanout")
            cost_train = (cost_train + 
                          (T.sum(T.mean(layers['dense_premax'].W, 
                                        axis=0)**2) * 
                           updatespecs['regularization']['l2_fanout']))
    
    print("\t\t...DONE")
    
    print("\tUPDATES...")
    
    # Fetch update parameters
    all_params = lasagne.layers.get_all_params(layers['output_layer'], trainable=True)
    
    # Create weight update function
    if updatespecs['alg'] == 'adadelta':
        lr = updatespecs.get('lr', 1.)
        updates = lasagne.updates.adadelta(cost_train, all_params, 
                                           learning_rate=lr)
    elif updatespecs['alg'] == 'nostr_mom':    
        momentum = updatespecs.get('momentum', 0.9)
        updates = lasagne.updates.nesterov_momentum(cost_train, all_params, 
                                      learning_rate=updatespecs['lr'], 
                                      momentum=momentum)
    elif updatespecs['alg'] == 'sgd':
        updates = lasagne.updates.sgd(cost_train, all_params, 
                                      learning_rate=updatespecs['lr'])
    elif updatespecs['alg'] == 'adam':
        lr = updatespecs.get('lr', 0.001)
        updates = lasagne.updates.adam(cost_train, all_params, 
                                      learning_rate=lr)
    else:
        exit("Specified update-algorithm not implemented!")
    
    # Optionally constrain weight values to position nodes
    constraint = updatespecs.get('constraint', False)
    if constraint:
        force_sum = updatespecs.get('force_sum', False)
        if layerspecs['input_encoder'] == 'triangles':
            # norm weights (triangles)
            updates[layers['dense_premax'].W] = \
                T.set_subtensor(\
                    updates[layers['dense_premax'].W][triangle_slice,:], \
                    constr_weight_sum(updates[layers['dense_premax'].W]\
                        [triangle_slice,:], max_sum=2., sum_axes=(0,),
                        force_sum=force_sum))
        elif layerspecs['input_encoder'] == 'binary':
            # norm weights (binary)
            updates[layers['dense_premax'].W] = \
                T.set_subtensor(\
                    updates[layers['dense_premax'].W][binary_slice,:], \
                    constr_weight_sum(updates[layers['dense_premax'].W]\
                        [binary_slice,:], max_sum=10., sum_axes=(0,),
                        force_sum=force_sum))
    
    # print number of params
    total_params = sum([p.get_value().size for p in all_params])
    print("\t\tNetwork params to train:", total_params)
    
    print("\t\t...DONE")
    
    
    # These lists specify that sym_input should take the value of sh_input and etc.
    # Note the cast: T.cast(sh_target, 'int32'). This is nessesary because Theano
    # does only support shared varibles with type float32. We cast the shared
    # value to an integer before it is used in the graph.
    givens = [(sym_input, sh_input),
              (sym_target, T.cast(sh_target, 'int32')),
              (sym_timesteps, T.cast(sh_timesteps, 'int32')),
              (sym_cost_weights, sh_cost_weights)]
    givens_preds = [(sym_input, sh_input), (sym_timesteps,
                                            T.cast(sh_timesteps, 'int32'))]
    
    # theano.function compiles a theano graph. [] means that the the function
    # takes no input because the inputs are specified with the givens argument.
    # We compile cost_train and specify that the parameters should be updated
    # using the update rules.
    print("COMPILING FUNCTIONS...\n\tTRAIN...", end=' ')
    
    train = theano.function([], cost_train, updates=updates, givens=givens,
                            on_unused_input=on_unused_input)#, mode=theano.compile.MonitorMode(post_func=detect_nan))#, mode="DebugMode")
    print("\tDONE")
    print("\tCOST-VAL...", end=' ')
    comp_cost = theano.function([], cost, givens=givens,
                                on_unused_input=on_unused_input)
    print("\tDONE")
    print("\tCOST-VAL...", end=' ')
    comp_cost_noreg = theano.function([], cost_noreg, givens=givens,
                                      on_unused_input=on_unused_input)
    print("\tDONE")
    print("\tPREDS...", end=' ')
    comp_preds = theano.function([], prediction, givens=givens_preds,
                                 on_unused_input=on_unused_input)
    print("\tDONE")
    
    return dict(
        train = train,
        cost = comp_cost,
        cost_noreg = comp_cost_noreg,
        pred = comp_preds,
        shared = dict(sh_input=sh_input, sh_target=sh_target,
                      sh_timesteps=sh_timesteps,
                      sh_cost_weights=sh_cost_weights)
        )


############################################################################
#  Plotting function                                                       #
############################################################################

def maxlayer_visualization(layers, layerspecs, name, comp_funcs, epoch,
                           samp_ids=None, minibatch=None, count=None,
                           markers=[], weights=True, plot_tc=True):
    """
    Plot TC layer activations over sequence positions and network weights, 
    by passing them to the plotting deamon
    
    Parameters
    -------
    weights : bool
        Plot network weights?
    plot_tc : bool
        Plot TC layer activations over sequence positions?
    see train() for usage of the other input parameters
    Notes
    -------
    Since maxlayer or meanlayer can be built, these layertypes are referred
    to as tc (timestep-convolution) layers.
    """
    
    if count == None:
        title = ", epoch {0:03d}".format(epoch)
        filename = "_ep{0:03d}".format(epoch)
    else:
        title = ", epoch {0:03d}/{1:03d}".format(epoch, count)
        filename = "_ep{0:03d}_mb{1:03d}".format(epoch, count)
    filename = filename + image_datatype
    
    # Plot weights
    if weights:
        #lasagne.layers.get_all_params() returns list of theano tensors
        # [W, W, b]
        all_params = lasagne.layers.get_all_params(layers['output_layer'])
        weights = [[all_params[0], "max_layer"]]
        if len(layerspecs['dense']) > 0:
            weights.append([all_params[-4], "dense layer"])
        
        weights.append([all_params[-2], 'output layer'])
        
        for weight in weights:
            
            plotting_queue.put([mypl.weight_heatmap, 
                                [dict(x=np.array(weight[0].eval()), 
                                      max_maplength=50, #wrap heatmaps
                                      title=weight[1]+title,
                                      xlabel="nodes",
                                      ylabel="inputs",
                                      savename=path.join(name,
                                                         weight[1]+filename)
                                      )]])
    
    
    # plot maxlayer activations
    if plot_tc:
        sh_input = comp_funcs['shared']['sh_input']
        sh_timesteps = comp_funcs['shared']['sh_timesteps']
        
        pred = comp_funcs['pred']()
        if len(layerspecs['dense']) > 0:
            importance = np.sum(np.abs(lasagne.layers.get_all_param_values(\
                                            layers['dense1'])[-2]), axis=1)
        else:
            importance = np.sum(np.abs(lasagne.layers.get_all_param_values(\
                                    layers['output_layer'])[-2]), axis=1)
        
        premax_preds = get_output(layers['dense_in'], 
                                  inputs=sh_input, 
                                  timesteps=sh_timesteps)[samp_ids,:].eval()
        premax_act = get_output(layers['premax_rs'], 
                                inputs=sh_input, 
                                timesteps=sh_timesteps)[samp_ids,:,:].eval()
        
        for s_i in samp_ids:
            suptitle =\
                "true: {}, pred: {},\n{}, {}".format(minibatch['y'][s_i],
                    pred[s_i], "maxlayer", minibatch['names'][s_i]) + title
            savename = path.join(name, "{}_{}".format("maxlayer", s_i) + 
                                 filename)
            
            plotting_queue.put([\
                mypl.timeseries_vs_preds_at_timesteps, 
                [dict(predictions=\
                            premax_act[s_i, :minibatch['last_ind'][s_i], :],
                      timesteps=\
                          minibatch['X_unenc'][s_i,
                                              :minibatch['last_ind'][s_i]],
                      impacts=premax_preds[s_i]*importance,
                      markers=markers,
                      suptitle=suptitle,
                      savename=savename
                      )]])


############################################################################
#  Network training loop                                                   #
############################################################################

def train(comp_funcs, dataset, reader, layers, layerspecs, num_epochs=None,
          name='net', plot=True):
    """
    Create and perform training loop (to be called in for-loop)
    
    Parameters
    -------
    
    comp_funcs : dict
        Theano functions created by create_iter_functions()
    dataset : dict
        A dataset dictionary as created by HDF5_Batch_Reader(), containing
        necessary information about the dataset sizes and dimensions
    reader : HDF5_Batch_Reader() instance
        Instance of minibatch reader class
    layers : dict
        Dictionary with lasagne layers as created by build_model()
    layerspecs : dict
        Specifications for the network design, such as layer sizes, dropout,
        etc.. See maxlayer_start.py for an example.
    num_epochs : int
        Number of taining epochs
    name : string
        Name of network (for plotting and saving trained network)
    plot : bool
        Enable plotting?
    Returns
    -------
    dict
        Dictionary with number of epoch, duration, and various performance
        measures
    """
    
    debug = False
    
    # Get shared theano variables in which minibatches will be stored
    sh_input = comp_funcs['shared']['sh_input']
    sh_target = comp_funcs['shared']['sh_target']
    sh_timesteps = comp_funcs['shared']['sh_timesteps']
    sh_cost_weights = comp_funcs['shared']['sh_cost_weights']
    
    
    ## Variables and arrays for scoring (AUC/BACC/loss)
    
    # Create a confusionmatrix for scoring for each class
    confmatrix_train = [ConfusionMatrix(2, flat=True)
                        for c in range(dataset['n_classes'])]
    confmatrix_valid = [ConfusionMatrix(2, flat=True)
                        for c in range(dataset['n_classes'])]
    
    # Dicts to store scores
    costs = dict(train = 0., valid = 0.) #current cost per minibatch
    best_AUC = dict(score=0., epoch=0) #best observed AUC
    best_BACC = dict(score=0., epoch=0) #best observed BACC
    best_error = dict(score=np.inf, epoch=0) #best observed loss
    
    # Preallocate arrays for AUC and BACC calculation
    # training set
    train_preds = np.zeros((np.ceil(float(len(dataset['train'])) / 
                                    dataset['batch_size']) * 
                            dataset['batch_size'], dataset['n_classes']), 
                           dtype=np.float64) #predictions
    train_trues = np.zeros_like(train_preds) #targets
    train_preds_classes = np.zeros_like(train_preds, dtype=np.int) #predcted classes
    train_weights = np.zeros_like(train_preds, dtype=np.bool) #sampleweights
    train_auc_scores = np.zeros(dataset['n_classes']) #AUC scores per class
    # validation set
    valid_preds = np.zeros((np.ceil(float(len(dataset['val'])) / 
                                    dataset['batch_size']) * 
                            dataset['batch_size'], dataset['n_classes']), 
                           dtype=np.float64) #predictions
    val_trues = np.zeros_like(valid_preds) #targets
    valid_preds_classes = np.zeros_like(valid_preds, dtype=np.int) #predcted classes
    valid_weights = np.zeros_like(valid_preds, dtype=np.bool) #sampleweights
    valid_auc_scores = np.zeros(dataset['n_classes']) #AUC scores per class
    # AUC scores per class for joined training and validation set
    total_auc_scores = np.zeros(dataset['n_classes'], dtype=np.float64)
    
    
    ## Training epochs
    
    # Plot initial weights
    if plot:
        maxlayer_visualization(layers=layers, layerspecs=layerspecs,
                               name=name, comp_funcs=comp_funcs,
                               samp_ids=None, minibatch=None, epoch=0,
                               count=None, markers=coding_snps,
                               plot_tc=False)
    # Loop containing training epochs
    for epoch in itertools.count(1):
        # print more information in first 2 epocs
        if (epoch <= 2):
            debug = True
        else:
            debug = False
        
        start_time = time.time()
        
        
        ## Training set
        
        # Reset confmatrix, costs, mb counter, and get number of minibatches
        [confm.zero() for confm in confmatrix_train]
        costs['train'] = 0.
        count = 1 #minibatch counter
        n_batches = np.ceil(float(len(dataset['train'])) / 
                            dataset['batch_size'])
        
        # Loop through minibatches
        for minibatch in minibatch_generator(sample_inds=dataset['train'],
                                        batch_size=dataset['batch_size'],
                                        samplefct=reader.load_data_mb,
                                        num_cached=5, rnd_gen=rnd_gen):
            
            if debug:
                print("  Processing minibatch  {0:4.0f} / {1:4.0f} (training) .".format(count, n_batches), end=' ')
                sys.stdout.flush()
            
            # Update value of shared variables to current minibatch
            sh_input.set_value(minibatch['X'], borrow=True)
            sh_target.set_value(minibatch['y'], borrow=True)
            sh_timesteps.set_value(minibatch['last_ind'], borrow=True)
            sh_cost_weights.set_value(minibatch['weights'], borrow=True)

            if debug:
                print(".", end=' ')
                sys.stdout.flush()
            
            # Calculate current cost and update weights
            cur_cost = float(comp_funcs['train']())
            
            if debug:
                print(".", end=' ')
                sys.stdout.flush()
            
            # Store costs, predictions, targets, and weights
            costs['train'] += cur_cost / n_batches
            train_preds[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] =comp_funcs['pred']()
            train_trues[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] = minibatch['y']
            train_weights[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] =minibatch['weights'] > 0
            
            # Option for plotting network during training
            if plot and debug and False:
                maxlayer_visualization(layers=layers, 
                                       layerspecs=layerspecs, name=name, 
                                       comp_funcs=comp_funcs, samp_ids=[0],
                                       minibatch=minibatch, epoch=epoch, 
                                       count=count, markers=coding_snps)
            
            # Print some info about minibatch training
            if debug:
                print("current loss: {0:12.8f}, at {1:12.5} sec\n".format(cur_cost, time.time()-start_time), end=' ')
                sys.stdout.flush()
            
            # Increment minibatch counter and delete minibatch from memory
            count += 1
            minibatch.clear()
            del minibatch
            
        if debug:
            print()
            sys.stdout.flush()
        
        
        ## Validation set
        
        # Reset confmatrix, costs, mb counter, and get number of minibatches
        [confm.zero() for confm in confmatrix_valid]
        costs['val'] = 0.
        count = 1 #minibatch counter
        n_batches = np.ceil(float(len(dataset['val'])) / 
                            dataset['batch_size'])

        # Loop through minibatches
        for minibatch in minibatch_generator(sample_inds=dataset['val'], 
                                        batch_size=dataset['batch_size'],
                                        samplefct=reader.load_data_mb, 
                                        num_cached=5, rnd_gen=rnd_gen):
            
            if debug:
                print("  Processing minibatch  {0:3.0f} / {1:3.0f} (validation) .".format(count,n_batches), end=' ')
            
            # Update value of shared variables to current minibatch
            sh_input.set_value(minibatch['X'], borrow=True)
            sh_target.set_value(minibatch['y'], borrow=True)
            sh_timesteps.set_value(minibatch['last_ind'], borrow=True)
            sh_cost_weights.set_value(minibatch['weights'], borrow=True)
            
            if debug:
                print(".", end=' ')
                sys.stdout.flush()
            
            # Calculate current cost (without updating weights)
            cur_cost = float(comp_funcs['cost']())
            
            if debug:
                print("current loss: {0:12.8f}, at {1:12.5}\r".format(cur_cost, time.time()-start_time), end=' ')
                sys.stdout.flush()
            
            # Store costs, predictions, targets, and weights
            costs['val'] += cur_cost / n_batches
            valid_preds[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] = comp_funcs['pred']()
            val_trues[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] = minibatch['y']
            valid_weights[(count-1)*dataset['batch_size']:\
                count*dataset['batch_size'],:] = minibatch['weights'] > 0
            
            # Plot network at first minibatch
            if plot and count==1 and ((epoch%plot_ep) == 0):
                maxlayer_visualization(layers=layers, layerspecs=layerspecs, 
                                       name=name, comp_funcs=comp_funcs, 
                                       samp_ids=[0], minibatch=minibatch,
                                       epoch=epoch, count=None, 
                                       markers=coding_snps)
            
            # Increment minibatch counter and delete minibatch from memory
            count += 1
            minibatch.clear()
            del minibatch
        
        if debug:
            print()
        
        # Round predictions to predicted classes
        train_preds_classes[:] = np.clip(np.rint(train_preds), 0, 1)
        valid_preds_classes[:] = np.clip(np.rint(valid_preds), 0, 1)
        
        
        ## Calculate Scores
        
        # Calculate AUC scores
        # training set
        train_auc_scores[:] = [roc_auc_score(\
                                    train_trues[train_weights[:, cl], cl],
                                    train_preds[train_weights[:, cl], cl],
                                    average=None)
                               for cl in range(dataset['n_classes'])]
        # validation set
        valid_auc_scores[:] = [roc_auc_score(\
                                    val_trues[valid_weights[:, cl], cl],
                                    valid_preds[valid_weights[:, cl], cl],
                                    average=None) 
                               for cl in range(dataset['n_classes'])]
        # joined training and validation set
        total_auc_scores[:] = [roc_auc_score(\
                                np.append(\
                                    train_trues[train_weights[:, cl], cl],
                                    val_trues[valid_weights[:, cl], cl]),
                                np.append(\
                                    train_preds[train_weights[:, cl], cl],
                                    valid_preds[valid_weights[:, cl], cl]),
                                average=None)
                               for cl in range(dataset['n_classes'])]
        # average AUCs over all classes
        train_auc = np.mean(train_auc_scores)
        valid_auc = np.mean(valid_auc_scores)
        total_auc = np.mean(total_auc_scores)
        
        # Save best validation AUC score and save network if new best
        if best_AUC['score'] <= np.mean(valid_auc_scores):
            best_AUC['score'] = np.mean(valid_auc_scores)
            best_AUC['epoch'] = epoch
            savemodel(layers['output_layer'],
                      filename=path.join(name, 'best_auc_net.p'))
            print("New best auc: epoch {}".format(epoch))
        
        # Calculate BACC scores
        # add training results to confusionmatrix (add all mbs at once)
        [confmatrix_train[cl].batchAdd(\
            train_trues[train_weights[:, cl].flatten(), cl],
            train_preds_classes[train_weights[:, cl].flatten(), cl])
         for cl in range(dataset['n_classes'])]
        # add validation results to confusionmatrix (add all mbs at once)
        [confmatrix_valid[cl].batchAdd(\
            val_trues[valid_weights[:, cl].flatten(), cl], 
            valid_preds_classes[valid_weights[:, cl].flatten(), cl]) 
         for cl in range(dataset['n_classes'])]
        # calculate TP, TN, FP, and FN rates
        rates_train = dict(zip(['tp','tn','fp','fn'], 
                               np.transpose(np.array([confm.getErrors() 
                                   for confm in confmatrix_train]))))
        rates_valid = dict(zip(['tp','tn','fp','fn'], 
                               np.transpose(np.array([confm.getErrors() 
                                   for confm in confmatrix_valid]))))
        # calculate BACC score
        train_baccs = np.array([confm.balanced_accuracy() 
                                for confm in confmatrix_train]).flatten()
        valid_baccs = np.array([confm.balanced_accuracy() 
                                for confm in confmatrix_valid]).flatten()
        # average BACCs over all classes
        train_bacc = np.mean(train_baccs)
        valid_bacc = np.mean(valid_baccs)
        
        # Save best validation BACC score and save network if new best
        if best_BACC['score'] <= valid_bacc:
            best_BACC['score'] = valid_bacc
            best_BACC['epoch'] = epoch
            savemodel(layers['output_layer'], 
                      filename=path.join(name, 'best_bacc_net.p'))
            print("New best bacc: epoch {}".format(epoch))
        
        # Save best validation loss and save network if new best
        if best_error['score'] >= costs['val']:
            best_error['score'] = costs['val']
            savemodel(layers['output_layer'], 
                      filename=path.join(name,'best_loss_net.p'))
            print("New best loss: epoch {}".format(epoch))
            best_error['epoch'] = epoch
        
        end_time = time.time()
        
        yield {
            'number': epoch,
            'duration': end_time-start_time,
            'costs': costs,
            
            'train_bacc': train_bacc,
            'valid_bacc': valid_bacc,
            
            'train_baccs': train_baccs,
            'valid_baccs': valid_baccs,
            
            'train_auc' : train_auc,
            'valid_auc' : valid_auc,
            'total_auc' : total_auc,
            
            'train_aucs' : train_auc_scores,
            'valid_aucs' : valid_auc_scores,
            'total_aucs' : total_auc_scores,
            
            'rates_train': rates_train,
            'rates_valid': rates_valid,
            'best_error' : best_error,
            'best_AUC' : best_AUC,
            'best_BACC' : best_BACC,
            }


def main(layerspecs, datafiles, num_epochs=10, batch_size=50, 
         updatespecs=None, loadname=None, name='net', all_reads=False,
         fold=0, folds=1):
    """
    Main function for creation of network and conducting training
    
    Parameters
    -------
    layerspecs : dict
        Specifications for the network design, such as layer sizes, dropout,
        etc.. See maxlayer_start.py for an example.
    datafiles : dict
        Dictionary with 'input_file' and 'target_file' keys. See
        maxlayer_start.py for an example.
    num_epochs : int
        Number of epochs to train
    batch_size : int
        Size of minibatches
    updatespecs : dict
        Dictionary with the update specifications and regularization
        methods.
    loadname : string
        Option for loading trained network from file
    name : string
        Name of network (for plotting and saving trained network)
    all_reads : bool
        True: Use minor and major SNVs as input sequence for samples;
        False: Use minor SNVs per sample only; Default: False
    fold : int
        Number of current cross-validation fold
    folds : int
        Total number of cross-validation folds
    Returns
    -------
    Prints results of training and saves trained networks
    """
    global coding_snps
    global rnd_gen
    
    
    ## Load data
    
    print("LOADING DATA...")
    
    # Extract coding SNV positions from filename (only for artificial 
    #  dataset)
    coding_snps = dict()
    if re.search('_v([0-9]*)',datafiles['input_file']) is not None:
        snp_var = int(re.search('_v([0-9]*)',
                                datafiles['input_file']).group(1))
        
        coding_snps['b'] = re.search('_p([0-9_]*)_a',
                                     datafiles['input_file']).group(1)
        if coding_snps['b'] == '':
            coding_snps['b'] = np.array([])
        else:
            if snp_var == 0:
                coding_snps['b'] = np.array(coding_snps['b'].split('_'),
                                            dtype=float)
            else:
                temp_snps = np.array(coding_snps['b'].split('_'),
                                     dtype=float)
                coding_snps['b'] = np.repeat(temp_snps, snp_var*2+1)
                coding_snps['b'] += np.tile(np.arange(snp_var*2+1),
                                            temp_snps.shape[0]) - snp_var
        
        coding_snps['r'] = re.search('_a([0-9_]*)_v',
                                     datafiles['input_file']).group(1)
        if coding_snps['r'] == '':
            coding_snps['r'] = np.array([])
        else:
            if snp_var == 0:
                coding_snps['r'] = np.array(coding_snps['r'].split('_'),
                                            dtype=float)
            else:
                temp_snps = np.array(coding_snps['r'].split('_'), 
                                     dtype=float)
                coding_snps['r'] = np.repeat(temp_snps, snp_var*2+1)
                coding_snps['r'] += np.tile(np.arange(snp_var*2+1), 
                                            temp_snps.shape[0]) - snp_var
    
    # Create instances of input encoder class (for mini-batch reader)
    if layerspecs['input_encoder'] == 'triangles':
        if layerspecs.get('nr_tria_nodes', 0):
            input_encoder = InputEncoderTriangles(ticker_steps=0, 
                nr_tria_nodes=layerspecs['nr_tria_nodes'], 
                add_binary_encoding=layerspecs.get('add_binary_encoding', 
                                                   True), 
                double_range=layerspecs.get('double_range', False))
        else:
            input_encoder = InputEncoderTriangles(ticker_steps=0, 
                min_dif=layerspecs['min_dif'], 
                add_binary_encoding=layerspecs.get('add_binary_encoding', 
                                                   True), 
                double_range=layerspecs.get('double_range', False))
    elif layerspecs['input_encoder'] == 'binary':
        input_encoder = InputEncoderBinary(ticker_steps=0)
    else:
        raise ValueError("Input encoder type not implemented!")
    
    # Create instances of minibatch-reader class
    reader = HDF5_Batch_Reader(datafiles['input_file'], 
                               datafiles['target_file'], batch_size,
                               input_encoder, ticker_steps=0, 
                               all_reads=all_reads, class_weights=True, 
                               targets=datafiles.get('targets', None),
                               verbose=True)
    dataset = reader.dataset
    
    
    # Create fold inidices by shuffling sample indices and
    #  splitting them evenly
    #get sample indices
    sample_inds = np.arange(len(reader.sample_names))
    
    #make mask for training set
    fold_points = np.linspace(0, len(sample_inds),folds+1)
    train_fold = np.ones(len(sample_inds), dtype=np.bool)
    train_fold[fold_points[fold]:fold_points[fold+1]] = 0
    
    #shuffle folds with constant random state
    rnd_gen = np.random.RandomState(seed=123)
    rnd_gen.shuffle(sample_inds)
    
    #initialize weights etc. with different state per fold
    try:
        rnd_gen = np.random.RandomState(123+int(123*fold/folds)+int(456*int(datafiles['target_file'][-1])/5))
    except ValueError:
        rnd_gen = np.random.RandomState(123+int(123*fold/folds))
    lasagne.random.set_rng(rnd_gen)
    
    #apply training-set mask to obtain indices in training or validation set
    dataset['train'] = sample_inds[train_fold]
    dataset['val'] = sample_inds[~train_fold]
    
    # Print a summary
    print("  Dataset: {}".format(datafiles))
    print("  Samples (train): {0}".format(len(dataset['train'])))
    print("  Samples (val): {0}".format(len(dataset['val'])))
    print("  Samples indices (val): {0}".format(dataset['val']))
    print("  Maximal value: {}".format(dataset['max_pos']))
    print("  Maximal length: {}".format(dataset['max_len']))
    print("  Ticker steps: {}".format(0))
    
    
    ## Build network
    
    print("BUILDING NETWORK...")
    
    # Build raw network
    layers = build_model(dataset=dataset, layerspecs=layerspecs, 
                         updatespecs=updatespecs)
    # Load a trained model
    if loadname != None:
        loadmodel(layers['output_layer'], loadname)
    
    # build theano functions
    comp_funcs = create_iter_functions(dataset=dataset, layers=layers, 
                                       updatespecs=updatespecs,
                                       layerspecs=layerspecs)
    
    
    ## Start training loop
    
    print("STARTING TRAINING...")
    
    # Initialize variables to check wether net is dead/finished
    best_val_cost = np.inf
    unchanged_cost_eps = 0
    max_unchanged_eps = num_epochs * 0.25
    
    # Start training
    start_time = time.time()
    for epoch in train(comp_funcs=comp_funcs, dataset=dataset, 
                       reader=reader, num_epochs=num_epochs, name=name, 
                       layers=layers, layerspecs=layerspecs, plot=False):
        # Print summary every epoch
        print("Epoch {0:d} of {1:d}".format(epoch['number'], num_epochs))
        print("  training loss:        {}".format(epoch['costs']['train']))
        print("  validation loss:      {}".format(epoch['costs']['val']))
        
        print("  avg. training BACC:   {} %".format(epoch['train_bacc'] * 100))
        print("  avg. validation BACC: {} %".format(epoch['valid_bacc'] * 100))
        
        print("  avg. training AUC:    {}".format(epoch['train_auc']))
        print("  avg. validation AUC:  {}".format(epoch['valid_auc']))
        print("  avg. total AUC:       {}".format(epoch['total_auc']))
        
        print("  training BACC:        {} %".format(epoch['train_baccs'] * 100))
        print("  validation BACC:      {} %".format(epoch['valid_baccs'] * 100))
        
        print("  training AUC:    {}".format(epoch['train_aucs']))
        print("  validation AUC:  {}".format(epoch['valid_aucs']))
        print("  total AUC:       {}".format(epoch['total_aucs']))
        
        print(("  Classification rates (train):"))
        for c in range(dataset['n_classes']):
            print(("  \tClass %d: FP: %d TP: %d FN: %d TN: %d") % (c, epoch['rates_train']['fp'][c], epoch['rates_train']['tp'][c], epoch['rates_train']['fn'][c], epoch['rates_train']['tn'][c]))
        print(("  Classification rates (valid):"))
        for c in range(dataset['n_classes']):
            print(("  \tClass %d: FP: %d TP: %d FN: %d TN: %d") % (c, epoch['rates_valid']['fp'][c], epoch['rates_valid']['tp'][c], epoch['rates_valid']['fn'][c], epoch['rates_valid']['tn'][c]))
        print(("  duration:              %.2f sec" % epoch['duration']))
        
        # Save model when plotting at an epoch
        if (epoch['number'] % plot_ep) == 0:
            savemodel(layers['output_layer'], filename=path.join(name,'ep'+str(epoch['number'])+'_'+'net.p'))
        
        # Stop training when desired number of epochs is reached
        if epoch['number'] >= num_epochs:
            break
        
        # Stop training if validation error does not improve for
        #  max_unchanged_eps epochs (early stopping)
        if (epoch['costs']['val'] - best_val_cost) >= 0:
            unchanged_cost_eps += 1
            if (unchanged_cost_eps > max_unchanged_eps) and (epoch['number'] > (num_epochs*0.15)):
                print("\tTraining stopped, validation error not improving for {} epochs.".format(max_unchanged_eps))
                break
        else:
            unchanged_cost_eps = 0
            best_val_cost = epoch['costs']['val']
            
            
    # When finished, save network
    savemodel(layers['output_layer'], filename=path.join(name,'fin'+'net.p'))
    print("\tResults saved!")
    
    # Print a small summary
    print(("\nTraining comlpete after %d epochs. Total duration: %.2f sec" % 
        (epoch['number'], time.time()-start_time)))
    
    print("\nBest validation error in epoch {} with {}".format(epoch['best_error']['epoch'], epoch['best_error']['score']))
    print("\nBest validation AUC in epoch {} with {}".format(epoch['best_AUC']['epoch'], epoch['best_AUC']['score']))
    print("\nBest validation BACC in epoch {} with {}".format(epoch['best_BACC']['epoch'], epoch['best_BACC']['score']))
    
    # Make sure to immediately free the memory
    del input_encoder, reader, comp_funcs, layers, dataset
    
    return 0

    
def start_experiment(exp_nr, num_epochs, batch_size, name, datafile, 
                     updatespec, layerspec, loadname, fold, 
                     folds, semaphore_pool, print_to_console):
    """
    Start an experiment by calling the main function.
    
    Parameters
    -------
    exp_nr : int
        Number or ID of experiment
    num_epochs : int
        Number of epochs to train
    batch_size : int
        Size of minibatches
    name : string
        Name of network (for plotting and saving trained network)
    datafiles : dict
        Dictionary with 'input_file' and 'target_file' keys. See
        maxlayer_start.py for an example.
    updatespecs : dict
        Dictionary with the update specifications and regularization
        methods.
    layerspecs : dict
        Specifications for the network design, such as layer sizes, dropout,
        etc.. See maxlayer_start.py for an example.
    loadname : string
        Option for loading trained network from file
    fold : int
        Number of current cross-validation fold
    folds : int
        Total number of cross-validation folds
    semaphore_pool : dict()
        'num_parallel' : multiprocessing.Semaphore() instance
            Semaphore to be aquired by experiments
    print_to_console : bool
        True: Print to console and write to file
        False: Only write to file
    Returns
    -------
    Prints results of training to console, writes to file, and saves trained
    networks
    """
    
    # Create logfile
    with gzip.open(name+".log.gz", "at") as log_file:
        # Redirect ouptput to file
        original_stdout = sys.stdout
        if print_to_console:
            sys.stdout = Tee(sys.stdout, original_stdout, log_file)
        else:
            sys.stdout = Tee(sys.stdout, log_file)
        
        print("Starting experiment {} ({})...".format(name, exp_nr))
        print("\tNet configuration: {}\n".format(layerspec))
        print("\tUpdate algorithm: {}\n".format(updatespec))
        print("\tMinibatch size: {}\n".format(batch_size))
        try:
            # Perform experiment
            main(updatespecs=updatespec, num_epochs=num_epochs,
                 datafiles=datafile, batch_size=batch_size,
                 layerspecs=layerspec, loadname=loadname, name=name,
                 all_reads=False, fold=fold, folds=folds)
            
            print("\nExperiment " + name +  " completed.\n\n" + "-"*25 + "\n\n")
        except Exception as e:
            # Terminate plotting demons and release semaphore if experiment
            #  failed
            print("\nExperiment failed!\n{}\n\n".format(e) + "-"*25 + "\n\n")
            if True:
                plotting_queue.put(0)
                print("Printing daemon termintated.")
                semaphore_pool['num_parallel'].release()
                raise
    
    # If finished, redirect sys.stdout to console
    sys.stdout = original_stdout
    
    # Immediately free memory
    gc.collect()
    gc.collect()
    gc.collect()
    
    # End and reap plotting deamons
    plotting_queue.put(0)
    
    if print_to_console:
        print("Printing daemon termintated.")
    
    # Release the semaphore
    semaphore_pool['num_parallel'].release()


if __name__ == '__main__':
    print("Please use this module via its start_experiment() function!")

