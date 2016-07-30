# -*- coding: utf-8 -*-
"""input_encoders_numba.py: Input encoder for positions


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

InputEncoderTriangles(): Input encoder class for triangle position 
    encodings, as described in thesis
InputEncoderBinary() Input encoder class for binary position encodings, as 
    described in thesis

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import numpy as np
import numba

def int_to_binary(int_array, exp):
    binary_array = (((int_array[:,np.newaxis].astype(int) & (1 << np.arange(exp)))) > 0).astype(int)
    return binary_array


def int_to_overlap_binary(int_array, exp):
    exp = int(np.ceil(exp))
    exps = (1 << np.arange(int(exp)))
    binary_array = np.zeros((int_array.shape[0],int(exp)*2-1))
    decimal_array = int_array[:,np.newaxis].astype(int)
    
    binary_array[:,:exp] =  ((decimal_array & exps) > 0).astype(int)
    
    shifted_decimal_array = np.zeros_like(decimal_array)
    for c in np.arange(exp-1):
        shifted_decimal_array[:] = decimal_array + exps[c]
        binary_array[:,exp+c] = ((shifted_decimal_array & exps) > 0).astype(int)[:,c+1]
    return binary_array

@numba.jit(signature_or_function='(int64, int64, int64, float64, int64, uint32[:], float32[:,:,:], uint32[:,:,:])',
           locals=dict(n_i=numba.int64, mask=numba.uint64, 
                       n_tria_nodes=numba.int64, s_i=numba.int64, 
                       t_i=numba.int64, remainder=numba.float32, 
                       quotient=numba.int64, orig_int=numba.uint64, 
                       pos_bit=numba.int64), nopython=True, nogil=True)
def encode_triangles_inplace_jit(n_binary_bits, ticker_steps, n_inputnodes, node_range, nr_samples, lengths, out, out_uint):
    # index at which to excpect position as integer
    pos_bit = n_inputnodes
    
    # structure: features, tickerstepnode, binarynodes, tianglenodes
    # use pos_bit for ticker_steps and move n_inputnodes
    if ticker_steps > 0:
        n_inputnodes -= 1
    
    # Mask for bit extraction
    mask = 1
    
    # Nodes for triangle encoding (see structure)
    n_tria_nodes = n_inputnodes - n_binary_bits
    
    for s_i in range(nr_samples):
        # Enocde position
        for t_i in range(lengths[s_i]):
            orig_int = out_uint[s_i, t_i, -pos_bit]
            
            # Add precise binary coded position information (do this first so orig_int can be deleted)
            for n_i in range(n_binary_bits):
                out[s_i, t_i, n_i-n_inputnodes] = ((mask << n_i) & orig_int) != 0
            
            
            # Encode position via triangles
            
            remainder = (orig_int % node_range) / node_range #activations from 0-1
            quotient = orig_int / node_range
            
            # use quotient as index and remainder as activation for left node
            out[s_i, t_i, quotient-n_tria_nodes] = 1. - remainder #activations from 0-1
            
            # use quotient+1 as index and remainder as activation for right node
            # wrap index around if (quotient+1)>n_tria_nodes
            quotient += 1
            if quotient-n_tria_nodes < 0:
                out[s_i, t_i, quotient-n_tria_nodes] = remainder
            else:
                out[s_i, t_i, -n_tria_nodes] = remainder
            
            remainder = remainder #activations from 0-1
            
            out_uint[s_i, t_i, -pos_bit] = 0 #remove position
            
        # Set ticker steps, if they exist
        if ticker_steps > 0:
            for n_i in range(ticker_steps):
                out[s_i, lengths[s_i]+n_i, -pos_bit] = 1
            # add ticker steps to lenght
            lengths[s_i] += n_i # := lengths[s_i] += lengths[s_i-1 for index of last element instead of length


@numba.jit(signature_or_function='(int64, int64, float64, int64, uint32[:], float32[:,:,:], uint32[:,:])',
           locals=dict(n_i=numba.int64, mask=numba.uint64, 
                       n_tria_nodes=numba.int64, s_i=numba.int64, 
                       t_i=numba.int64, remainder=numba.float32, 
                       quotient=numba.int64, 
                       pos_bit=numba.int64), nopython=True, nogil=True)
def encode_triangles_jit(n_binary_bits, n_inputnodes, node_range, nr_samples, lengths, out, positions):
    # structure: binarynodes, tianglenodes
    
    # Mask for bit extraction
    mask = 1
    
    for s_i in range(nr_samples):
        # Enocde position
        for t_i in range(lengths[s_i]):
            # Add precise binary coded position information
            for n_i in range(n_binary_bits):
                out[s_i, t_i, n_i-n_inputnodes] = ((mask << n_i) & positions[s_i, t_i]) != 0
            
            
            # Encode position via triangles
            
            remainder = (positions[s_i, t_i] % node_range) / node_range #activations from 0-1
            quotient = positions[s_i, t_i] / node_range
            
            # use quotient as index and 1-remainder as activation for left node
            quotient = quotient + n_binary_bits
            out[s_i, t_i, quotient] = 1. - remainder #activations from 0-1
            
            # use quotient+1 as index and remainder as activation for right node
            # wrap index around if (quotient+1)>n_tria_nodes
            quotient += 1
            out[s_i, t_i, quotient] = remainder
            

@numba.jit(signature_or_function='(int64, int64, int64, float64, int64, uint32[:], float32[:,:,:], uint32[:,:])',
           locals=dict(n_i=numba.int64, mask=numba.uint64,
                       n_tria_nodes=numba.int64, s_i=numba.int64, 
                       t_i=numba.int64, remainder=numba.float32, 
                       quotient=numba.int64, 
                       pos_bit=numba.int64), nopython=True, nogil=True)
def encode_double_triangles_jit(n_binary_bits, n_inputnodes, tria_nodes_end, node_range, nr_samples, lengths, out, positions):
    # structure: binarynodes, tianglenodes
    
    # Mask for bit extraction
    mask = 1
    
    for s_i in range(nr_samples):
        # Enocde position
        for t_i in range(lengths[s_i]):
            # Add precise binary coded position information
            for n_i in range(n_binary_bits):
                out[s_i, t_i, n_i-n_inputnodes] = ((mask << n_i) & positions[s_i, t_i]) != 0
            
            
            # Encode position via triangles
            
            remainder = (positions[s_i, t_i] % node_range) / node_range #activations from 0-1
            quotient = positions[s_i, t_i] / node_range
            
            # use quotient as index and 1-remainder as activation for left node
            quotient = quotient + n_binary_bits
            out[s_i, t_i, quotient] = ((1. - remainder) / 4.) + 0.25 #activations from 0-1
            
            if quotient > 0: #check if there is a quotient-1 node available
                if (out[s_i, t_i, quotient] - 0.25 > 0):
                    out[s_i, t_i, quotient-1] = out[s_i, t_i, quotient] - 0.25
            
            # use quotient+1 as index and remainder as activation for right node
            # wrap index around if (quotient+1)>n_tria_nodes
            quotient += 1
            out[s_i, t_i, quotient] = (remainder / 4.) + 0.25
            
            if quotient < tria_nodes_end: #check if there is a quotient-1 node available
                if (out[s_i, t_i, quotient] - 0.25 > 0):
                    out[s_i, t_i, quotient+1] = out[s_i, t_i, quotient] - 0.25
            


class InputEncoderTriangles(object):
    def __init__(self, ticker_steps=0, tstep_resolution=1.0, min_dif=None, node_range=None, nr_tria_nodes=None, add_binary_encoding=True, double_range=False): 
        
        if min_dif != None:
            self.node_range = np.ceil(np.float64(tstep_resolution) / min_dif) # half the (symmetric) range of the node
        elif node_range != None:
            self.node_range = np.ceil(np.float64(node_range) / 2.)
        elif nr_tria_nodes != None:
            self.node_range = None
            self.nr_tria_nodes = nr_tria_nodes
        else:
            raise(ValueError, "min_dif or node_range have to be specified!")
        
        self.ticker_steps = np.int64(ticker_steps)
        self.add_binary_encoding = add_binary_encoding
        self.double_range = double_range
        
    def calculate_parameters(self, max_val):
        if self.node_range == None:
            self.node_range = np.ceil(np.float64(max_val) / self.nr_tria_nodes)
        if self.add_binary_encoding:
            self.n_binary_bits = np.int64(np.ceil(np.log2(self.node_range))) #encode one triangle range binary for more precision
        else:
            self.n_binary_bits = 0
        self.helper_nodes = (self.ticker_steps > 0)
        
        self.max_val = np.float64(max_val)
        self.data_nodes = np.ceil(self.max_val / self.node_range) + 1  + self.n_binary_bits # +1 for right side node (no wrap-around)
        self.n_inputnodes =  np.int64(self.data_nodes + self.helper_nodes)
        
    def encode_inplace_jit(self, nr_samples, lengths, out):
        """ Need to use this function to call the jit compiled function, as
            class support via numba is disabled sice 0.12"""
        encode_triangles_inplace_jit(n_binary_bits=self.n_binary_bits,
                             ticker_steps=self.ticker_steps, 
                             n_inputnodes=self.n_inputnodes,
                             node_range=self.node_range,
                             nr_samples=nr_samples, lengths=lengths, out=out, 
                             out_uint=out.view(np.uint32))
    
    def encode_jit(self, nr_samples, lengths, out, positions):
        """ out[samples, reads, bin_nodes+tria_nodes]
            Need to use this function to call the jit compiled function, as
            class support via numba is disabled sice 0.12"""
        if self.double_range:
            encode_double_triangles_jit(n_binary_bits=np.int64(self.n_binary_bits),
                                 n_inputnodes=np.int64(self.data_nodes),
                                 tria_nodes_end=self.data_nodes-self.n_binary_bits-1, #this is the last tria index -> nr nodes - 1
                                 node_range=np.float64(self.node_range),
                                 nr_samples=np.int64(nr_samples),
                                 lengths=np.uint32(lengths), out=np.float32(out), 
                                 positions=np.uint32(positions))
        else:
            encode_triangles_jit(n_binary_bits=np.int64(self.n_binary_bits),
                                 n_inputnodes=np.int64(self.data_nodes),
                                 node_range=np.float64(self.node_range),
                                 nr_samples=np.int64(nr_samples),
                                 lengths=np.uint32(lengths), out=np.float32(out), 
                                 positions=np.uint32(positions))
               
    
    def encode(self, inp, lengths):
        #input shape: minibatchsize x input_len
        #output shape: minibatchsize x input_len x n_inputnodes
        minibatchsize = inp.shape[0]
        output = np.zeros((minibatchsize, inp.shape[1]+self.ticker_steps, self.n_inputnodes), dtype=float)
        lengths -= (self.ticker_steps - 1)
        
        for mb in np.arange(minibatchsize):
            
            scaled_pos = inp[mb,:] / self.node_range
            # equals output[np.arange(len(input)), np.trunc(scaled_pos)+1] except for the last timestep
            output[mb, np.arange(inp.shape[1]), scaled_pos.astype(int)+(scaled_pos.astype(int)<self.data_nodes)] = np.abs(self.max_act * (scaled_pos-np.trunc(scaled_pos)))
            output[mb, np.arange(inp.shape[1]), scaled_pos.astype(int)] = np.abs(self.max_act - output[mb, np.arange(inp.shape[1]), scaled_pos.astype(int)+(scaled_pos.astype(int)<self.data_nodes)])
            output[mb, np.arange(inp.shape[1]), -self.exp-1:-1] = int_to_binary(inp[mb,:] % self.node_range, self.exp)
            if self.ticker_steps > 0:
                output[mb, lengths[mb]:, :] = 0
                output[mb, lengths[mb]:lengths[mb]+self.ticker_steps, -1] = 1
        
        return output


@numba.jit(signature_or_function='(int32, int64, int64, uint32[:], float32[:,:,:], uint32[:,:,:])', 
     locals=dict(s_i=numba.int64, t_i=numba.int64, n_i=numba.int64, mask=numba.uint64, orig_int=numba.uint64, pos_bit=numba.int64, n_bits=numba.uint64), nopython=True, nogil=True)
def encode_overlapping_binary_inplace_jit(ticker_steps, n_inputnodes, nr_samples, lengths, out, out_uint):
    # out: float32[:,:,:] with [samples, timesteps, features]
    # pos_ind: int64 at which position in out['features'] the position
    #    is stored and encoding should start
    # lengths: lengths without tickersteps - tickersteps will be added here
    # Note: positions=0 are ignored
    mask = 1
    pos_bit = n_inputnodes
    if ticker_steps > 0:
        n_inputnodes -= 1
    n_bits = (n_inputnodes / 2) + 1 #lowbit only in binary
    
    for s_i in range(nr_samples):
        
        # Enocde position
        for t_i in range(lengths[s_i]):
            orig_int = out_uint[s_i, t_i, -pos_bit] #to be encoded
            
            # normal bitwise for all bits
            for n_i in range(n_bits):
                out[s_i, t_i, n_i-n_inputnodes] = ((mask << n_i) & orig_int) != 0
            
            # overlapping bitwise (low bit ignored)
            for n_i in range(1, n_bits):
                # for bit at bitposition bp add 2**(bp-1) = 1<<(bp-1)
                out[s_i, t_i, n_i-n_bits] = ((mask << n_i) & (orig_int + (mask << (n_i-1)))) != 0
            
            out[s_i, t_i, -pos_bit] = 0 #remove position
        
        
        # Set ticker steps, if they exist
        if ticker_steps > 0:
            for n_i in range(ticker_steps):
                out[s_i, lengths[s_i]+n_i, -pos_bit] = 1
            # add ticker steps to lenght
            lengths[s_i] += n_i # := lengths[s_i] += lengths[s_i-1 for index of last element instead of length

@numba.jit(signature_or_function='(int64, int64, uint32[:], float32[:,:,:], uint32[:,:])', 
     locals=dict(s_i=numba.int64, t_i=numba.int64, n_i=numba.int64, mask=numba.uint64, orig_int=numba.uint64, pos_bit=numba.int64, n_bits=numba.uint64), nopython=True, nogil=True)
def encode_overlapping_binary_jit(n_inputnodes, nr_samples, lengths, out, positions):
    # out: float32[:,:,:] with [samples, timesteps, features]
    # pos_ind: int64 at which position in out['features'] the position
    #    is stored and encoding should start
    # lengths: lengths without tickersteps - tickersteps will be added here
    mask = 1
    n_bits = (n_inputnodes / 2) + 1 #lowbit only in binary
    
    for s_i in range(nr_samples):
        
        # Enocde position
        for t_i in range(lengths[s_i]):
            # normal bitwise for all bits
            for n_i in range(n_bits):
                out[s_i, t_i, n_i-n_inputnodes] = ((mask << n_i) & positions[s_i, t_i]) != 0
            
            # overlapping bitwise (low bit ignored)
            for n_i in range(1, n_bits):
                # for bit at bitposition bp add 2**(bp-1) = 1<<(bp-1)
                out[s_i, t_i, n_i-n_bits] = ((mask << n_i) & (positions[s_i, t_i] + (mask << (n_i-1)))) != 0
            
        

class InputEncoderBinary(object):
    def __init__(self, ticker_steps=0):
        self.ticker_steps = np.int64(ticker_steps)
        if self.ticker_steps > 0:
            self.helper_nodes = 1
        else:
            self.helper_nodes = 0
        
        
    def calculate_parameters(self, max_val):
        
        self.exp = np.int64(np.ceil(np.log2(max_val)))
        self.data_nodes = self.exp * 2 - 1
        
        self.n_inputnodes =  np.int64(self.data_nodes + self.helper_nodes)
        self.max_val = max_val
        
    def encode_inplace_jit(self, nr_samples, lengths, out):
        """ Need to use this function to call the jit compiled function, as
            class support via numba is disabled sice 0.12"""
        encode_overlapping_binary_inplace_jit(ticker_steps=self.ticker_steps, 
                             n_inputnodes=self.n_inputnodes,
                             nr_samples=nr_samples, lengths=lengths, out=out, 
                             out_uint=out.view(np.uint32))
        
    def encode_jit(self, nr_samples, lengths, out, positions):
        """ Need to use this function to call the jit compiled function, as
            class support via numba is disabled sice 0.12"""
        encode_overlapping_binary_jit(n_inputnodes=self.data_nodes,
                             nr_samples=nr_samples, lengths=lengths, out=out, 
                             positions=np.uint32(positions))
        
    def encode(self, inp, lengths, output=None):
        #input shape: minibatchsize x input_len
        #output shape: minibatchsize x input_len x n_inputnodes
        vector_flag = False
        if len(inp.shape) <= 1:
            vector_flag = True
            inp = inp[:,np.newaxis]
        minibatchsize = inp.shape[0]
        if output is None:
            output = np.zeros((minibatchsize, inp.shape[1]+self.ticker_steps, self.n_inputnodes), dtype=np.float32)
        lengths -= (self.ticker_steps - 1)
        
        for mb in np.arange(minibatchsize):
            
            output[mb, np.arange(inp.shape[1]), :self.data_nodes] = int_to_overlap_binary(inp[mb,:], self.exp)
            if self.ticker_steps > 0:
                output[mb, lengths[mb]:, :] = 0
                output[mb, lengths[mb]:lengths[mb]+self.ticker_steps, -1] = 1
        if vector_flag:
            return output[:,0,:]
        else:
            return output
        