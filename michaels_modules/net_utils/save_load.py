# -*- coding: utf-8 -*-
"""save_load.py: Functions for saving/loading trained networks


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Convenience functions for saving/loading trained networks (wrapping lasagne
functions lasagne.layers.set_all_param_values() and 
lasagne.layers.set_all_param_values())

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import pickle as pickle
import gzip

def savemodel(layer, filename, verbose=True):
    import lasagne.layers
    if verbose:
        print('Saving net...', end=' ')
    params = lasagne.layers.get_all_param_values(layer)
    with gzip.open(filename+'.gz', 'wb') as f:
        pickle.dump(params, f,
            protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('DONE')

def loadmodel(layer, filename, verbose=True):
    import lasagne.layers
    if verbose:
        print('Loading net...', end=' ')
    
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            params = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            params = pickle.load(f)
    
    lasagne.layers.set_all_param_values(layer, params)
    if verbose:
        print('DONE')

def load_model_params(filename, verbose=True):
    if verbose:
        print('Loading net...', end=' ')
    
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            params = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            params = pickle.load(f)
    
    if verbose:
        print('DONE')
    return params