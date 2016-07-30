# -*- coding: utf-8 -*-
"""nonlinearities.py: Nonlinearities used in MSc thesis


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Various custom and modified activation functions. See my thesis for more
detailed descriptions. (Not all functions actually used in thesis.)

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""


import theano.tensor as T
import lasagne.nonlinearities

def relu_cap(x):
    return T.switch(T.switch(x>1, 1, x)<0, 0, x)

def rectify_leaky_cap(leakyness=0.01, caps=[0.,1.]):
    assert leakyness < 1 and leakyness > 0, "leakyness should be ]0-1["
    return lambda x: T.switch(T.switch(x>caps[1], x*leakyness, x)<caps[0], x*leakyness, x)

def rectify(offset=0.):
    #return lambda x: 0.5 * (x + abs(x)) # This faster approach results in calculation errors at some points (nans)
    if offset == 0.:
        return lambda x: T.maximum(0, x)
    else:
        return lambda x: T.maximum(0, (x + offset))

def sigmoid(offset=0.):
#    return lambda x: 0.5 * (x + abs(x)) # This faster approach results in calculation errors at some points (nans)
    if offset == 0.:
        return lambda x: lasagne.nonlinearities.sigmoid(x)
    else:
        return lambda x: lasagne.nonlinearities.sigmoid(x - offset)

def rectify_tanh(offset=0.):
    """rectified tanh, optional offset giving max(0, tanh(x+offset))
    """
    if offset == 0.:
        return lambda x: T.switch(T.gt(x,0), lasagne.nonlinearities.tanh(x), 0)
    else:
        return lambda x: T.switch(T.gt((x + offset),0), lasagne.nonlinearities.tanh(x + offset), 0)

def elu(alpha=1., offset=0.):
    """ elu activation as proposed by [1]_
    
    References
    -------
    .. [1] Clevert, D., Unterthiner, T. & Hochreiter, S., 2015. "Fast and 
        accurate deep network learning by exponential linear units (elus)". 
        CoRR, abs/1511.07289. Available at: 
        <http://arxiv.org/abs/1511.07289> [Accessed 21/07/2016]
    """
    if offset == 0.:
        return lambda x: T.switch(T.gt(x,0), x, alpha*(T.exp(x)-1))
    else:
        return lambda x: T.switch(T.gt((x + offset),0), (x + offset), alpha*(T.exp((x + offset))-1))

def etanh(alpha=1., offset=0.):
    if offset == 0.:
        return lambda x: lasagne.nonlinearities.tanh(T.switch(T.gt(x,0), x, alpha*(T.exp(x)-1)))
    else:
        return lambda x: lasagne.nonlinearities.tanh(T.switch(T.gt((x + offset),0), (x + offset), alpha*(T.exp((x + offset))-1)))

def erth(alpha=1., offset=0.):
    if offset == 0.:
        return lambda x: T.switch(T.gt(x,0), lasagne.nonlinearities.tanh(x), alpha*(T.exp(lasagne.nonlinearities.tanh(x))-1))
    else:
        return lambda x: T.switch(T.gt((x + offset),0), (lasagne.nonlinearities.tanh(x + offset)), alpha*(T.exp((lasagne.nonlinearities.tanh(x + offset)))-1))

def static_out(val=1.):
    return lambda x: val
    
def log_softmax(x):
    """ Stable theano log_softmax
    
    Somehow theano currently only optimizes for stable integer vectors, when
    using imatrix, NaNs might be generated.
    Code by f0k in
    https://github.com/Lasagne/Lasagne/issues/332#issuecomment-122328992
    """
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets):
    """ Stable theano categorical_crossentropy_logdomain
    
    Somehow theano currently only optimizes for stable integer vectors, when
    using imatrix, NaNs might be generated.
    Code by f0k in
    https://github.com/Lasagne/Lasagne/issues/332#issuecomment-122328992
    """
    return -T.sum(targets * log_predictions, axis=1)