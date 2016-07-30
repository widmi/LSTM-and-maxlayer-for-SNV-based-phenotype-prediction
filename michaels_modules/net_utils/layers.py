# -*- coding: utf-8 -*-
"""layers.py: Different layer types and modifications used in MSc thesis


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Various custom and modified lasagne layers. Most layers are based on
existing lasagne layers and only contain some modifications.
=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""
from collections import OrderedDict

import theano
import theano.tensor as T
import lasagne.layers
from lasagne.utils import unroll_scan
from lasagne import nonlinearities, init
from lasagne.layers import Gate
import numpy as np

class OneTimestepLayer(lasagne.layers.Layer):
    """
    Only forward outputs at certain/last sequence positions
    
    Parameters
    -------
    input_layer : lasagne layer type
    Input layer with shape: [samples, sequence positions, features]
    """
    
    def __init__(self, input_layer):
        # Initialize parent layer
        super(OneTimestepLayer, self).__init__(input_layer)
        self.input_shape = lasagne.layers.get_output_shape(input_layer)
        
    def get_output_for(self, input, timesteps=None, *args, **kwargs):
        """
        Only forward outputs at certain/last sequence positions
        
        Parameters
        -------
        input : tensor
            Input layer with shape: [samples, sequence positions, features]
        timesteps : array of integers or None
            None: Take output at last sequence position
            Array of integers: take outputs at sequence positions specified
                in array; values serve as indices and must not exeed
                sequence lenght; length of array must be number of samples
        """
        if timesteps != None:
            return input[T.arange(start=0,stop=self.input_shape[0]),timesteps,:]
        else:
            return input[:,-1,:]
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


class MaxLayer(lasagne.layers.Layer):
    """
    Calculate maximum activation over activations at sequence positions per
    sample
    
    Parameters
    -------
    input_layer : lasagne layer type
        Input layer
    axis: int
        Axis to calculate maximum over (default: -1, all axes)
    keepdims : bool
        Keep Number of dimensions?
    """
    def __init__(self, input_layer, axis=-1, keepdims=False):
        # Initialize parent layer
        super(MaxLayer, self).__init__(input_layer)
        self.keepdims = keepdims
        self.axis = axis
    
    def get_output_for(self, input, *args, **kwargs):
        return T.max(input, axis=self.axis, keepdims=self.keepdims)
    
    def get_output_shape_for(self, input_shape):
        output_shape = np.array(input_shape)
        if self.keepdims:
            return tuple(output_shape)
        else:
            return tuple(np.delete(output_shape, self.axis))

class MeanLayer(lasagne.layers.Layer):
    """
    Calculate mean activation over activations at sequence positions per
    sample
    
    Parameters
    -------
    input_layer : lasagne layer type
        Input layer
    axis: int
        Axis to calculate mean over (default: -1, all axes)
    keepdims : bool
        Keep Number of dimensions?
    """
    def __init__(self, input_layer, axis=-1, keepdims=False):
        # Initialize parent layer
        super(MeanLayer, self).__init__(input_layer)
        self.keepdims = keepdims
        self.axis = axis
    
    def get_output_for(self, input, *args, **kwargs):
        return T.mean(input, axis=self.axis, keepdims=self.keepdims)
    
    def get_output_shape_for(self, input_shape):
        output_shape = np.array(input_shape)
        if self.keepdims:
            return tuple(output_shape)
        else:
            return tuple(np.delete(output_shape, self.axis))

class InputDropoutLayer(lasagne.layers.DropoutLayer):
    """
    Apply input dropout consistently over sequence positions as suggested by 
    Gal (2015) http://arxiv.org/pdf/1512.05287.pdf
    
    Parameters
    -------
    incoming : lasagne layer type
        Input layer
    p: float
        Dropout probability
    rescale : bool
        Rescale not dropped out activations by 1/p?
    """
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(InputDropoutLayer, self).__init__(incoming, **kwargs)
        self.p = p
        self.rescale = rescale
        constant_dim = 1
        self.constant_dim = constant_dim
        self.dropoutshape = (self.input_shape[:constant_dim]) + (self.input_shape[constant_dim+1:])
        self.dropoutlayer = lasagne.layers.DropoutLayer(incoming=self.dropoutshape, p=p, rescale=rescale, **kwargs)
        
        # add parameters to this layer
        self.params.update(self.dropoutlayer.params)
        
        self.dropoutmask = self.add_param(init.Constant(1), 
                                          self.dropoutshape, 'dropoutmask',
                                          trainable=False, regularizable=False)
        
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled
        """
        if deterministic or (self.p == 0):
            return input
        else:
            dropoutmask = self.dropoutlayer.get_output_for(self.dropoutmask, **kwargs)
            input *= dropoutmask.dimshuffle(0, 'x', 1)
            
            return input

class LSTMLayerTransparent(lasagne.layers.LSTMLayer):

    r"""
    Modified version of lasagne LSTM layer class. Includes cell state cap, 
    LSTM Batch-Normalization on forward and recurrent connections at
    multiple parts of the LSTM, and LSTM recurrent dropout
    Also provides get_states_for() function, which acts like
    the standard get_output_for() but also returns the LSTM states in the
    LSTM (gate activations, cell state, Batch-Normalization parameters,
    etc.) for closer analysis.
    
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    cell_state_cap : int, tuple of integers, None
        Clip the cell state at +/- int or use the integers in the tuple as
        minimum and maximum
    batchnorm : dict or False
        LSTM Batch-Normalilzation as suggested by [2]_
        
        False: 
            No Batch-Normalization.
        dict : 
            'epsilon', 'alpha', 'beta', 'gamma', 'mean', 'inv_std' :
            initializations (see in paper);
            'bn_input', 'bn_cell_output', 'bn_cell_state', 'bn_cell_input', 
            'bn_ingate', 'bn_outgate', 'bn_forgetgate', 
            'bn_cell_input_nonlin' : bool
            True: Apply Batch-Normalization to this LSTM part; 
            'bn_cell_output' includes recurrent connections in LSTM;
            as in my thesis, I suggest dict(bn_input=True, 
            bn_cell_output) (default: False)
    lstm_rec_dropout : dict, bool, or None
        Dropout on LSTM recurrent connections as suggested by [3]_
        
        None : 
            No recurrent dropout
        True :
            Use standard settings for dropout
        dict :
            'p': dropout probability (default: 0.25);
            'rescale': rescale remaining connections by 1/p? (default: True)
    References
    ----------
    .. [1] Graves, A., 2013a. "Generating sequences with recurrent neural 
        networks". CoRR, abs/1308.0850. Available at: 
        <http://arxiv.org/abs/1308.0850> [Accessed 30/04/2016].

    .. [2] Cooijmans, T., Ballas, N., Laurent, C. & Courville, A.: 2016. 
        "Recurrent batch normalization". arXiv preprint, arXiv:1603.09025. 
        Available at: <http://arxiv.org/abs/1603.09025> 
        [Accessed 16/04/2016].
    .. [3] Gal, Y., 2015. "A theoretically grounded application of dropout 
        in recurrent neural networks". arXiv preprint, arXiv:1512.05287. 
        Available at: <http://arxiv.org/pdf/1512.05287.pdf> 
        [Accessed 26/07/2016].


    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 cell_state_cap=None,
                 batchnorm=False,
                 lstm_rec_dropout=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, lasagne.layers.base.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, lasagne.layers.base.Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(lasagne.layers.LSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.cell_state_cap = cell_state_cap
        self.batchnorm = batchnorm
        
        if self.cell_state_cap != None:
            if not isinstance(self.cell_state_cap, tuple):
                self.cell_state_cap = tuple([-self.cell_state_cap, self.cell_state_cap])
            elif len(self.cell_state_cap) == 1:
                self.cell_state_cap = tuple([-self.cell_state_cap[0], self.cell_state_cap[0]])
        
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0] #(batch_size, timesteps, features)
        
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])
        
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, lasagne.layers.base.Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, lasagne.layers.base.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        
        if isinstance(self.batchnorm, dict):
            # Apply batchnormalization on cell input (before addition and input nonlinearity)
            # TODO: (this is a marker)
            # Since the distribution of the foward inputs and reccurent inputs differs, BN will be applied to the forward inputs and to each timestep of the reccurent inputs
            
            # Fetch batchnorm user parameters (defaults according to http://arxiv.org/abs/1603.09025) except for alpha/epsilon
            epsilon = self.batchnorm.get('epsilon', 0.001)
            alpha = self.batchnorm.get('alpha', 0.1)
            beta = self.batchnorm.get('beta', init.Constant(0))
            gamma = self.batchnorm.get('gamma', init.Constant(0.1))
            mean = self.batchnorm.get('mean', init.Constant(0))
            inv_std = self.batchnorm.get('inv_std', init.Constant(1))
            
            bn_keys = ['bn_input', 'bn_cell_output', 'bn_cell_state', 'bn_cell_input', 'bn_ingate', 'bn_outgate', 'bn_forgetgate', 'bn_cell_input_nonlin']
            bns = [self.batchnorm.get(bn_key, False) != False for bn_key in bn_keys]
            
            # create OrderedDict with existing BN variations and make an index for them [self.bn_apps['bn_cell_input']]
            self.bns = OrderedDict([(bn[0], bn[2]) for bn in zip(bn_keys, bns, np.cumsum(bns)-1) if bn[1]])
            
            # Some shapes
            #batch_size = input_shape[0]
            timesteps = input_shape[1]
            #input_feats = input_shape[2]
            
            # One batchnorm per timestep and lstm unit -> each timestep and lstm unit needs its own params, plus 1 dimension for possible multiple BNs
            
            
            self.bn_means = list()
            self.bn_inv_stds = list()
            self.bn_betas = list()
            self.bn_gammas = list()
            
            for bn in self.bns.keys():
                if bn == 'bn_input':
                    batchnorm_param_shape = tuple([timesteps, 4*num_units])
                else:
                    batchnorm_param_shape = tuple([timesteps, num_units])
                self.bn_means.append(self.add_param(beta, batchnorm_param_shape, 'mean_'+bn,
                                           trainable=False, regularizable=False))
                self.bn_inv_stds.append(self.add_param(gamma, batchnorm_param_shape, 'inv_std_'+bn,
                                            trainable=False, regularizable=False))
                
                
                # the learnable parameters will be stored in one large 3D matrix
                self.bn_betas.append(self.add_param(mean, batchnorm_param_shape, 'beta_'+bn,
                                           trainable=True, regularizable=False))
                self.bn_gammas.append(self.add_param(inv_std, batchnorm_param_shape, 'gamma_'+bn,
                                              trainable=True, regularizable=True))

                
            self.epsilon = epsilon
            self.alpha = alpha
            self.min_samples_bn_stat = 5 # only update the BN statistics at the current timestep if more than 5 samples have the timeteps
        
        self.lstm_rec_dropout = lstm_rec_dropout
        if lstm_rec_dropout is not None:
            # Apply dropout to the recurrent connections according to 
            # http://arxiv.org/abs/1512.05287 by dropping out the 
            # recurrent weights, default values also from paper
            
            # convenience option, set lstm_rec_dropout=True for default vals
            if lstm_rec_dropout == True:
                lstm_rec_dropout = dict()
            
            # set the defaults
            lstm_rec_dropout['p'] = lstm_rec_dropout.get('p', 0.25)
            lstm_rec_dropout['rescale'] = lstm_rec_dropout.get('rescale', True)
            
            # w_hid shape is (num_units, 4*num_units)
            w_hid_shape = (num_units, 4*num_units)
            
            # create dropout layer
            self.lstm_rec_dropout = lasagne.layers.DropoutLayer(incoming=w_hid_shape, **lstm_rec_dropout)
            
            # add parameters to this layer
            self.params.update(self.lstm_rec_dropout.params)
                
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units
    
    def get_output_for(self, inputs, deterministic=False, 
                       batch_norm_use_averages=None, 
                       batch_norm_update_averages=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        
        # Apply dropout to hidden weights
        if self.lstm_rec_dropout is not None:
            W_hid_stacked = self.lstm_rec_dropout.get_output_for(W_hid_stacked, deterministic)
        
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
        
        # Decide about which behaviours to set for BN
        if isinstance(self.batchnorm, dict):
            # Decide whether to use the stored averages or mini-batch statistics
            if batch_norm_use_averages is None:
                batch_norm_use_averages = deterministic
            use_averages = batch_norm_use_averages
            
            # Decide whether to update the stored averages
            if batch_norm_update_averages is None:
                batch_norm_update_averages = not deterministic
            update_averages = batch_norm_update_averages
            
        
        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        
        
        # Convenience function for extraction of arguments in step_bn_masked
        def slice_args(a, start, multiplier, offset=0):
            return a[offset+start*multiplier:offset+(start+1)*multiplier]
        
        
        ## Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        ## Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            
            return [cell, hid]
        
        
        def apply_bn(data, bn_type, mean, inv_std, gamma, beta, prev_mean, prev_inv_std):
            # mean and inv_std are the running averages, use/update them if specified
            # take mean and inv_std over all samples
            cur_mean = data.mean(axis=0)
            cur_inv_std = T.inv(T.sqrt(data.var(axis=0) + self.epsilon))
            
            if update_averages:
                # update statistics
                mean[self.bns[bn_type]] = (1 - self.alpha) * mean[self.bns[bn_type]] + self.alpha * cur_mean
                inv_std[self.bns[bn_type]] = (1 - self.alpha) * inv_std[self.bns[bn_type]] + self.alpha * cur_inv_std
            if use_averages:
                # perform batchnorm
                data = (data - mean[self.bns[bn_type]]) * (gamma[self.bns[bn_type]] * inv_std[self.bns[bn_type]]) + beta[self.bns[bn_type]]
            else:
                # perform batchnorm
                data = (data - cur_mean) * (gamma[self.bns[bn_type]] * cur_inv_std) + beta[self.bns[bn_type]]
            
            return data
        
        
        ## Create single recurrent computation step function with batchnorm
        # input_n is the n'th vector of the input
        def step_bn(input_n, *args):
            args = list(args)
            mean = slice_args(args, 0, len(self.bns))
            inv_std = slice_args(args, 1, len(self.bns))
            beta = slice_args(args, 2, len(self.bns))
            gamma = slice_args(args, 3, len(self.bns))
            
            cell_previous = args[4*len(self.bns)]
            hid_previous = args[4*len(self.bns)+1]
            
            prev_mean = slice_args(args, 0, len(self.bns), offset=4*len(self.bns)+2)
            prev_inv_std = slice_args(args, 1, len(self.bns), offset=4*len(self.bns)+2)
            
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # BN over all fwd gates and cell input
            if 'bn_input' in self.bns:
                input_n = apply_bn(input_n, 'bn_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Extract the pre-activation gate values for the foward connections
            ingate = slice_w(input_n, 0)
            forgetgate = slice_w(input_n, 1)
            cell_input = slice_w(input_n, 2)
            outgate = slice_w(input_n, 3)
            
            # Specific BNs
            if 'bn_ingate' in self.bns:
                ingate = apply_bn(ingate, 'bn_ingate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_forgetgate' in self.bns:
                forgetgate = apply_bn(forgetgate, 'bn_forgetgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_cell_input' in self.bns:
                cell_input = apply_bn(cell_input, 'bn_cell_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_outgate' in self.bns:
                outgate = apply_bn(outgate, 'bn_outgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Calculate recurrent gate pre-activations
            input_rec = T.dot(hid_previous, W_hid_stacked)
            # Extract the pre-activation gate values for the recurrent connections and add them to the foward activations
            ingate += slice_w(input_rec, 0)
            forgetgate += slice_w(input_rec, 1)
            cell_input += slice_w(input_rec, 2)
            outgate += slice_w(input_rec, 3)
            
            # Clip gradients
            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(
                    ingate, -self.grad_clipping, self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(
                    forgetgate, -self.grad_clipping, self.grad_clipping)
                cell_input = theano.gradient.grad_clip(
                    cell_input, -self.grad_clipping, self.grad_clipping)
                outgate = theano.gradient.grad_clip(
                    outgate, -self.grad_clipping, self.grad_clipping)
            
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            
            if 'bn_cell_input_nonlin' in self.bns:
                cell_input = apply_bn(cell_input, 'bn_cell_input_nonlin', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
                
            outgate = self.nonlinearity_outgate(outgate)

            if 'bn_cell_output' in self.bns:
                cell_norm = apply_bn(cell, 'bn_cell_output', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell_norm)
                
            elif 'bn_cell_state' in self.bns:
                cell = apply_bn(cell, 'bn_cell_state', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            else:
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            
            return [cell, hid, mean, inv_std]
        
        
        def apply_bn_masked(data, mask, bn_type, mean, inv_std, gamma, beta, prev_mean, prev_inv_std):
            # mask=(timesteps, n_batches, 1), cell_input=(timesteps, n_batches, n_units), mean/inv_std=(timesteps, n_units)->(timesteps, 'x', n_units)
            # get statistics at this timestep, input_n=(n_time_steps, n_batch, 4*num_units)
            masked_data = data[mask.nonzero(return_matrix=False)[0],:]
            # Note: we could use theano's lazy ifelse here since the condition is a scalar but somehow it's faster with switch
            cur_mean = T.switch(T.sum(mask)>self.min_samples_bn_stat, masked_data.mean(axis=0, keepdims=False), prev_mean[self.bns[bn_type]])
            
            unstable_inv_std = T.sqrt(masked_data.var(axis=0, keepdims=False))
            cur_inv_std = T.switch(T.sum(mask)>self.min_samples_bn_stat, T.inv(T.switch(unstable_inv_std, unstable_inv_std, self.epsilon)), prev_inv_std[self.bns[bn_type]])
            
            # mean and inv_std are the running averages, use/update them if specified
            if update_averages:
                # update statistics
                mean[self.bns[bn_type]] = (1 - self.alpha) * mean[self.bns[bn_type]] + self.alpha * cur_mean
                inv_std[self.bns[bn_type]] = (1 - self.alpha) * inv_std[self.bns[bn_type]] + self.alpha * cur_inv_std
            if use_averages:
                # perform batchnorm
                data = (data - mean[self.bns[bn_type]]) * (gamma[self.bns[bn_type]] * inv_std[self.bns[bn_type]]) + beta[self.bns[bn_type]]
            else:
                # perform batchnorm
                data = (data - cur_mean) * (gamma[self.bns[bn_type]] * cur_inv_std) + beta[self.bns[bn_type]]
            
            return data
        
        ## Create single recurrent computation step function with batchnorm and mask
        # input_n is the n'th vector of the input
        def step_bn_masked(input_n, mask_n, *args):
            args = list(args)
            mean = slice_args(args, 0, len(self.bns))
            inv_std = slice_args(args, 1, len(self.bns))
            beta = slice_args(args, 2, len(self.bns))
            gamma = slice_args(args, 3, len(self.bns))
            
            cell_previous = args[4*len(self.bns)]
            hid_previous = args[4*len(self.bns)+1]
            
            prev_mean = slice_args(args, 0, len(self.bns), offset=4*len(self.bns)+2)
            prev_inv_std = slice_args(args, 1, len(self.bns), offset=4*len(self.bns)+2)
            
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # BN over all fwd gates and cell input
            if 'bn_input' in self.bns:
                input_n = apply_bn_masked(input_n, mask_n, 'bn_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Extract the pre-activation gate values for the foward connections
            ingate = slice_w(input_n, 0)
            forgetgate = slice_w(input_n, 1)
            cell_input = slice_w(input_n, 2)
            outgate = slice_w(input_n, 3)
            
            # Specific BNs
            if 'bn_ingate' in self.bns:
                ingate = apply_bn_masked(ingate, mask_n, 'bn_ingate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_forgetgate' in self.bns:
                forgetgate = apply_bn_masked(forgetgate, mask_n, 'bn_forgetgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_cell_input' in self.bns:
                cell_input = apply_bn_masked(cell_input, mask_n, 'bn_cell_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_outgate' in self.bns:
                outgate = apply_bn_masked(outgate, mask_n, 'bn_outgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Calculate recurrent gate pre-activations
            input_rec = T.dot(hid_previous, W_hid_stacked)
            # Extract the pre-activation gate values for the recurrent connections and add them to the foward activations
            ingate += slice_w(input_rec, 0)
            forgetgate += slice_w(input_rec, 1)
            cell_input += slice_w(input_rec, 2)
            outgate += slice_w(input_rec, 3)
            
            # Clip gradients
            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(
                    ingate, -self.grad_clipping, self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(
                    forgetgate, -self.grad_clipping, self.grad_clipping)
                cell_input = theano.gradient.grad_clip(
                    cell_input, -self.grad_clipping, self.grad_clipping)
                outgate = theano.gradient.grad_clip(
                    outgate, -self.grad_clipping, self.grad_clipping)
            
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            
            if 'bn_cell_input_nonlin' in self.bns:
                cell_input = apply_bn_masked(cell_input, mask_n, 'bn_cell_input_nonlin', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            
            if 'bn_cell_output' in self.bns:
                cell_norm = apply_bn_masked(cell, mask_n, 'bn_cell_output', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell_norm)
                
            elif 'bn_cell_state' in self.bns:
                cell = apply_bn_masked(cell, mask_n, 'bn_cell_state', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            else:
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            
            return [cell, hid] + mean + inv_std
        
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            if isinstance(self.batchnorm, dict):
                sequences = sequences + self.bn_means + self.bn_inv_stds + self.bn_betas + self.bn_gammas
                step_fun = step_bn_masked
            else:
                step_fun = step_masked
        else:
            sequences = [input]
            if isinstance(self.batchnorm, dict):
                sequences = sequences + [self.mean, self.inv_std, self.beta, self.gamma]
                step_fun = step_bn
            else:
                step_fun = step
        
        
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.base.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.base.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        
        # Prepare outputs_info for scan
        if isinstance(self.batchnorm, dict):
            if mask is not None:
                outputs_info = [cell_init, hid_init] + [m[0,:] for m in self.bn_means] + [i[0,:] for i in self.bn_inv_stds]
            else:
                outputs_info = [cell_init, hid_init, None, None]
        else:
            outputs_info = [cell_init, hid_init]
        
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            scan_returns = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            scan_returns, _ = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)
                
        if isinstance(self.batchnorm, dict):
            [cell_out, hid_out] = scan_returns[:2]
            means = slice_args(scan_returns, 0, len(self.bns), 2)
            inv_stds = slice_args(scan_returns, 1, len(self.bns), 2)
            
            if update_averages:
                for bn in range(len(self.bns)):
                    running_mean = theano.clone(self.bn_means[bn], share_inputs=False)
                    running_inv_std = theano.clone(self.bn_inv_stds[bn], share_inputs=False)
                    # set a default update for them:
                    running_mean.default_update = means[bn]
                    running_inv_std.default_update = inv_stds[bn]
                    hid_out += 0 * running_mean
                    hid_out += 0 * running_inv_std
        else:
            [cell_out, hid_out] = scan_returns
        
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
    
    
    def get_states_for(self, inputs, deterministic=False, 
                       batch_norm_use_averages=None, 
                       batch_norm_update_averages=None, 
                       return_forward_inputs=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : dictionary of theano.TensorType elements
            States for LSTM cell, output, cell_input, ingate, outgate, and
            Batch-Normalization values packed in a dictionary.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Apply dropout to hidden weights
        if self.lstm_rec_dropout is not None:
            W_hid_stacked = self.lstm_rec_dropout.get_output_for(W_hid_stacked, deterministic)
            
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
            
        if isinstance(self.batchnorm, dict):
            # Decide whether to use the stored averages or mini-batch statistics
            if batch_norm_use_averages is None:
                batch_norm_use_averages = deterministic
            use_averages = batch_norm_use_averages
            
            # Decide whether to update the stored averages
            if batch_norm_update_averages is None:
                batch_norm_update_averages = not deterministic
            update_averages = batch_norm_update_averages
            
        
        # At each call to scan, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        
        
        # Convenience function for extraction of arguments in step_bn_masked
        def slice_args(a, start, multiplier, offset=0):
            return a[offset+start*multiplier:offset+(start+1)*multiplier]
        
        
        ## Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, cell_input, ingate, outgate]
        
        ## Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            
            return [cell, hid, cell_input, ingate, outgate]
        
        
        def apply_bn(data, bn_type, mean, inv_std, gamma, beta, prev_mean, prev_inv_std):
            # mean and inv_std are the running averages, use/update them if specified
            # take mean and inv_std over all samples
            cur_mean = data.mean(axis=0)
            cur_inv_std = T.inv(T.sqrt(data.var(axis=0) + self.epsilon))
            
            if update_averages:
                # update statistics
                mean[self.bns[bn_type]] = (1 - self.alpha) * mean[self.bns[bn_type]] + self.alpha * cur_mean
                inv_std[self.bns[bn_type]] = (1 - self.alpha) * inv_std[self.bns[bn_type]] + self.alpha * cur_inv_std
            if use_averages:
                # perform batchnorm
                data = (data - mean[self.bns[bn_type]]) * (gamma[self.bns[bn_type]] * inv_std[self.bns[bn_type]]) + beta[self.bns[bn_type]]
            else:
                # perform batchnorm
                data = (data - cur_mean) * (gamma[self.bns[bn_type]] * cur_inv_std) + beta[self.bns[bn_type]]
            
            return data
        
        
        ## Create single recurrent computation step function with batchnorm
        # input_n is the n'th vector of the input
        def step_bn(input_n, *args):
            args = list(args)
            mean = slice_args(args, 0, len(self.bns))
            inv_std = slice_args(args, 1, len(self.bns))
            beta = slice_args(args, 2, len(self.bns))
            gamma = slice_args(args, 3, len(self.bns))
            
            cell_previous = args[4*len(self.bns)]
            hid_previous = args[4*len(self.bns)+1]
            
            prev_mean = slice_args(args, 0, len(self.bns), offset=4*len(self.bns)+2)
            prev_inv_std = slice_args(args, 1, len(self.bns), offset=4*len(self.bns)+2)
            
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # BN over all fwd gates and cell input
            if 'bn_input' in self.bns:
                input_n = apply_bn(input_n, 'bn_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Extract the pre-activation gate values for the foward connections
            ingate = slice_w(input_n, 0)
            forgetgate = slice_w(input_n, 1)
            cell_input = slice_w(input_n, 2)
            outgate = slice_w(input_n, 3)
            
            # Specific BNs
            if 'bn_ingate' in self.bns:
                ingate = apply_bn(ingate, 'bn_ingate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_forgetgate' in self.bns:
                forgetgate = apply_bn(forgetgate, 'bn_forgetgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_cell_input' in self.bns:
                cell_input = apply_bn(cell_input, 'bn_cell_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_outgate' in self.bns:
                outgate = apply_bn(outgate, 'bn_outgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Calculate recurrent gate pre-activations
            input_rec = T.dot(hid_previous, W_hid_stacked)
            # Extract the pre-activation gate values for the recurrent connections and add them to the foward activations
            ingate += slice_w(input_rec, 0)
            forgetgate += slice_w(input_rec, 1)
            cell_input += slice_w(input_rec, 2)
            outgate += slice_w(input_rec, 3)
            
            # Clip gradients
            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(
                    ingate, -self.grad_clipping, self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(
                    forgetgate, -self.grad_clipping, self.grad_clipping)
                cell_input = theano.gradient.grad_clip(
                    cell_input, -self.grad_clipping, self.grad_clipping)
                outgate = theano.gradient.grad_clip(
                    outgate, -self.grad_clipping, self.grad_clipping)
            
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            
            
            if 'bn_cell_input_nonlin' in self.bns:
                cell_input = apply_bn(cell_input, 'bn_cell_input_nonlin', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
                
            outgate = self.nonlinearity_outgate(outgate)

            if 'bn_cell_output' in self.bns:
                cell_norm = apply_bn(cell, 'bn_cell_output', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell_norm)
                
            elif 'bn_cell_state' in self.bns:
                cell = apply_bn(cell, 'bn_cell_state', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            else:
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            return [cell, hid, mean, inv_std, cell_input, ingate, outgate]
        
        
        def apply_bn_masked(data, mask, bn_type, mean, inv_std, gamma, beta, prev_mean, prev_inv_std):
            # mask=(timesteps, n_batches, 1), cell_input=(timesteps, n_batches, n_units), mean/inv_std=(timesteps, n_units)->(timesteps, 'x', n_units)
            # get statistics at this timestep, input_n=(n_time_steps, n_batch, 4*num_units)
            masked_data = data[mask.nonzero(return_matrix=False)[0],:]
            cur_mean = T.switch(T.sum(mask)>self.min_samples_bn_stat, masked_data.mean(axis=0, keepdims=False), prev_mean[self.bns[bn_type]])
            
            unstable_inv_std = T.sqrt(masked_data.var(axis=0, keepdims=False))
            cur_inv_std = T.switch(T.sum(mask)>self.min_samples_bn_stat, T.inv(T.switch(unstable_inv_std, unstable_inv_std, self.epsilon)), prev_inv_std[self.bns[bn_type]])
            
            # mean and inv_std are the running averages, use/update them if specified
            if update_averages:
                # update statistics
                mean[self.bns[bn_type]] = (1 - self.alpha) * mean[self.bns[bn_type]] + self.alpha * cur_mean
                inv_std[self.bns[bn_type]] = (1 - self.alpha) * inv_std[self.bns[bn_type]] + self.alpha * cur_inv_std
            if use_averages:
                # perform batchnorm
                data = (data - mean[self.bns[bn_type]]) * (gamma[self.bns[bn_type]] * inv_std[self.bns[bn_type]]) + beta[self.bns[bn_type]]
            else:
                # perform batchnorm
                data = (data - cur_mean) * (gamma[self.bns[bn_type]] * cur_inv_std) + beta[self.bns[bn_type]]
            
            return data
        
        
        ## Create single recurrent computation step function with batchnorm and mask
        # input_n is the n'th vector of the input
        def step_bn_masked(input_n, mask_n, *args):
            args = list(args)
            mean = slice_args(args, 0, len(self.bns))
            inv_std = slice_args(args, 1, len(self.bns))
            beta = slice_args(args, 2, len(self.bns))
            gamma = slice_args(args, 3, len(self.bns))
            
            cell_previous = args[4*len(self.bns)]
            hid_previous = args[4*len(self.bns)+1]
            
            prev_mean = slice_args(args, 0, len(self.bns), offset=4*len(self.bns)+2)
            prev_inv_std = slice_args(args, 1, len(self.bns), offset=4*len(self.bns)+2)
            
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            
            # BN over all fwd gates and cell input
            if 'bn_input' in self.bns:
                input_n = apply_bn_masked(input_n, mask_n, 'bn_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Extract the pre-activation gate values for the foward connections
            ingate = slice_w(input_n, 0)
            forgetgate = slice_w(input_n, 1)
            cell_input = slice_w(input_n, 2)
            outgate = slice_w(input_n, 3)
            
            # Specific BNs
            if 'bn_ingate' in self.bns:
                ingate = apply_bn_masked(ingate, mask_n, 'bn_ingate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_forgetgate' in self.bns:
                forgetgate = apply_bn_masked(forgetgate, mask_n, 'bn_forgetgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_cell_input' in self.bns:
                cell_input = apply_bn_masked(cell_input, mask_n, 'bn_cell_input', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            if 'bn_outgate' in self.bns:
                outgate = apply_bn_masked(outgate, mask_n, 'bn_outgate', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
            
            # Calculate recurrent gate pre-activations
            input_rec = T.dot(hid_previous, W_hid_stacked)
            # Extract the pre-activation gate values for the recurrent connections and add them to the foward activations
            ingate += slice_w(input_rec, 0)
            forgetgate += slice_w(input_rec, 1)
            cell_input += slice_w(input_rec, 2)
            outgate += slice_w(input_rec, 3)
            
            # Clip gradients
            if self.grad_clipping:
                ingate = theano.gradient.grad_clip(
                    ingate, -self.grad_clipping, self.grad_clipping)
                forgetgate = theano.gradient.grad_clip(
                    forgetgate, -self.grad_clipping, self.grad_clipping)
                cell_input = theano.gradient.grad_clip(
                    cell_input, -self.grad_clipping, self.grad_clipping)
                outgate = theano.gradient.grad_clip(
                    outgate, -self.grad_clipping, self.grad_clipping)
            
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            
            if 'bn_cell_input_nonlin' in self.bns:
                cell_input = apply_bn_masked(cell_input, mask_n, 'bn_cell_input_nonlin', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            
            if self.cell_state_cap != None:
                cell = T.clip(cell, self.cell_state_cap[0], self.cell_state_cap[1])
            
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            
            if 'bn_cell_output' in self.bns:
                cell_norm = apply_bn_masked(cell, mask_n, 'bn_cell_output', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell_norm)
                
            elif 'bn_cell_state' in self.bns:
                cell = apply_bn_masked(cell, mask_n, 'bn_cell_state', mean, inv_std, gamma, beta, prev_mean, prev_inv_std)
                
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            else:
                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
            
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            
            return [cell, hid] + mean + inv_std + [cell_input, ingate, outgate]
        
        
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            if isinstance(self.batchnorm, dict):
                sequences = sequences + self.bn_means + self.bn_inv_stds + self.bn_betas + self.bn_gammas
                step_fun = step_bn_masked
            else:
                step_fun = step_masked
        else:
            sequences = [input]
            if isinstance(self.batchnorm, dict):
                sequences = sequences + self.bn_means + self.bn_inv_stds + self.bn_betas + self.bn_gammas
                step_fun = step_bn
            else:
                step_fun = step
        
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.base.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.base.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        
        # Prepare outputs_info for scan
        if isinstance(self.batchnorm, dict):
            if mask is not None:
                outputs_info = [cell_init, hid_init] + [m[0,:] for m in self.bn_means] + [i[0,:] for i in self.bn_inv_stds] + [None, None, None]
            else:
                outputs_info = [cell_init, hid_init, None, None, None]
        else:
            outputs_info = [cell_init, hid_init, None, None, None]
        
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            scan_returns = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            scan_returns, _ = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)
        
        if isinstance(self.batchnorm, dict):
            [cell_out, hid_out] = scan_returns[:2]
            means = slice_args(scan_returns, 0, len(self.bns), 2)
            inv_stds = slice_args(scan_returns, 1, len(self.bns), 2)
            [cell_input, ingate, outgate] = scan_returns[-3:]
            
            if update_averages:
                for bn in range(len(self.bns)):
                    running_mean = theano.clone(self.bn_means[bn], share_inputs=False)
                    running_inv_std = theano.clone(self.bn_inv_stds[bn], share_inputs=False)
                    # set a default update for them:
                    running_mean.default_update = means[bn]
                    running_inv_std.default_update = inv_stds[bn]
                    hid_out += 0 * running_mean
                    hid_out += 0 * running_inv_std
                
                    
        else:
            [cell_out, hid_out, cell_input, ingate, outgate] = scan_returns
        
        
        # Give back activations at all timesteps and dimshuffle back to (n_batch, n_time_steps, n_features))
        lstm_states = dict(zip(['cell', 'output', 'cell_input', 'ingate', 'outgate'], [s.dimshuffle(1, 0, 2) for s in [cell_out, hid_out, cell_input, ingate, outgate]]))

        
        if isinstance(self.batchnorm, dict):
            for bn in range(len(self.bns)):
                lstm_states[str(self.bn_means[bn])] = self.bn_means[bn]
                lstm_states[str(self.bn_inv_stds[bn])] = self.bn_inv_stds[bn]
                lstm_states[str(self.bn_betas[bn])] = self.bn_betas[bn]
                lstm_states[str(self.bn_gammas[bn])] = self.bn_gammas[bn]
        
        if return_forward_inputs:
            # (n_time_steps, n_batch, 4*num_units) -> (n_batch, n_time_steps, 4*num_units)
            input = input.dimshuffle(1, 0, 2)
            def slice_whole_w(x, n):
                return x[:, :, n*self.num_units:(n+1)*self.num_units]
            lstm_states['forward_ingate'] = slice_whole_w(input, 0)
            lstm_states['forward_cell_input'] = slice_whole_w(input, 2)
        
        # if scan is backward reverse the output
        if self.backwards:
            for s in ['cell', 'output', 'cell_input', 'ingate', 'outgate']:
                lstm_states[s] = lstm_states[s][:, ::-1]
        
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        return lstm_states


class InitExcl(lasagne.init.Initializer):
    """Sample initial weights from a sampler sampler, exluding given
    indices, which are set to 0.
    Parameters
    ----------
    sampler : lasagne.init.Initializer instance
        Initialization values will be sampled from here
    excl : array of integers
        Indices for values to set to 0
    incl: array of integers
        If specified, only initialize the values at the indices in incl and
        set all others to 0
    """
    def __init__(self, sampler, excl=[], incl=[]):
        self.sampler = sampler
        self.excl = np.array(excl, dtype=np.int)
        self.incl = np.array(incl, dtype=np.int)
    def sample(self, shape):
        if len(self.incl):
            self.excl = np.arange(shape[0])
            self.excl = np.delete(self.excl, self.incl)
        samp = self.sampler.sample(shape=shape)
        samp[self.excl,:] = 0.
        return samp

class InitAbs(lasagne.init.Initializer):
    """Sample initial weights from the uniform distribution, exluding given indices.
    """
    def __init__(self, sampler):
        self.sampler = sampler
    def sample(self, shape):
        samp = self.sampler.sample(shape=shape)
        return np.abs(samp)

class InitExclAbs(lasagne.init.Initializer):
    """Convert initialization values to absolute values
    """
    def __init__(self, sampler, excl=[], incl=[]):
        self.sampler = sampler
        self.excl = np.array(excl, dtype=np.int)
        self.incl = np.array(incl, dtype=np.int)
    def sample(self, shape):
        if len(self.incl):
            self.excl = np.arange(shape[0])
            self.excl = np.delete(self.excl, self.incl)
        samp = self.sampler.sample(shape=shape)
        samp[self.excl,:] = 0.
        return np.abs(samp)

class InitArray(lasagne.init.Initializer):
    """Initialize weights with constant value.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, array):
        self.array = np.array(array)

    def sample(self, shape):
        return lasagne.utils.floatX(self.array)
