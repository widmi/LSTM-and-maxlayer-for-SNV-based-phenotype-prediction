# -*- coding: utf-8 -*-
"""regularization.py: Weight constraint for norming LSTM weights


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Weight constraint for norming LSTM weights (not used in my thesis but
available as regularization option).

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""


import theano
import theano.tensor as T

import numpy as np

def constr_weight_sum(tensor_var, max_sum=1., sum_axes=None, epsilon=1e-7, 
                      force_sum=True):
    """
    Constraint sum of absolute weights to maximum value max_sum by 
    downscaling all weights if force_sum=False or force sum of absolute
    weight values to max_sum if force_sum=True
    
    """
    
    ndim = tensor_var.ndim
    
    if sum_axes is not None:
        sum_over = tuple(sum_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(theano.config.floatX).type
    real_sums = T.sum(T.abs_(tensor_var), axis=sum_over, keepdims=True)
    if force_sum:
        target_sums = dtype(max_sum)
    else:
        target_sums = T.clip(real_sums, 0, dtype(max_sum))
    
    constrained_output = \
        (tensor_var * (target_sums / (dtype(epsilon) + real_sums)))
    
    return constrained_output
