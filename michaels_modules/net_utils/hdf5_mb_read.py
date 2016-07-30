# -*- coding: utf-8 -*-
"""hdf5_mb_read.py: Data reader


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

minibatch_generator(): Generates minibatches in threads via 
    HDF5_Batch_Reader
HDF5_Batch_Reader(): Class for reading data into mini-batches for training

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import numpy as np
import time
import queue
import threading
import h5py
import pandas as pa
#numba.compiler.config.OPT = 3 #somehow this is faster than '3' oO
#import pyximport; pyximport.install()
#import mb_input_decoder
from numba import jit
import numba

def minibatch_generator(sample_inds, batch_size, samplefct, num_cached=5, 
                        rnd_gen=None, verbose=False):
    """
    Spawns background-threads to read minibatches via load_data_mb() like
    function in HDF5_Batch_Reader() class. Inspired by Jan Schlueter (f0k).
    
    Parameters
    -------
    sample_inds : list or np.array of integers
        Indices of samples to read (indices refer to the position in data
        file)
    batch_size : int
        Size of minibatches
    samplefct : function
        Function to read minibatches like load_data_mb() in 
        HDF5_Batch_Reader() class
    num_cached : int
        Number of minibatches to cache
    rnd_gen : numpy random generator
        Optionally, a random generator can be provided for shuffling of the
        sample indices across the minibatches
    Returns
    -------
    Minibatch as returned by samplefct
    """
    
    mb_queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference
    
    # shuffle batches across minibatches
    if verbose:
        print("  Shuffling samples...", end=" ")
    
    if rnd_gen == None:
        np.random.shuffle(sample_inds)
    else:
        rnd_gen.shuffle(sample_inds)
    
    if verbose:
        print("DONE")
    
    n_batches = np.ceil(float(len(sample_inds)) / batch_size)
    minibatch_slices = [slice(i * batch_size, (i + 1) * batch_size)
                        for i in np.arange(n_batches)]
    
    def producer():
        for mb_sl in minibatch_slices:
            mb_samples = sample_inds[mb_sl]
            minibatch = samplefct(mb_samples, verbose=verbose)
            mb_queue.put(minibatch)
        mb_queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read minibatches from mb_queue, in current thread)
    minibatch = mb_queue.get()
    while minibatch is not sentinel:
        yield minibatch
        minibatch.clear()
        del minibatch
        mb_queue.task_done()
        minibatch = mb_queue.get()


class HDF5_Batch_Reader(object):
    def __init__(self, input_file, target_file, batch_size, input_encoder, 
                 dataset=dict(), all_reads=False, ticker_steps=0,
                 verbose=True, in_memory=True, class_weights=False, 
                 targets=None):
        """
        Reader and parser for hdf5 input files
        
        Parameters
        -------
        input_file : string
            Location of hdf5 input file with SNV information
        target_file : string
            Location of tab-separated csv target file containing sample classes.
            First column should contain sample names in first column with othe
            columns starting with the class name followed with the 0 to 1
            relation of the samples to the class with -1 for unknown relations.
        batch_size : int
            Size of minibatches
        input_encoder : function
            Function to encode inputs
        dataset : dict
            Optionally pre-existing dataset dictionary
        all_reads : bool
            Not implemented in this version!
            True: Use all SNVs for samples (minor and major)
            False: Only use minor SNVs per sample
        ticker_steps : int
            Number of ticker steps (for LSTM)
        in_memory : bool
            True: Read complete input file into memory
            False: Read on-the-fly only
        class_weights : bool
            Apply weighting of samples by ratio of number of samples in classes?
        targets : list of strings or None
            Specify classes to use.
            None: Take second column in target file as class to consider
            list of strings: Search for strings in firs row and only consider
            these columns
        
        Returns
        -------
        No returns but class properties like dataset dictionary
        """
        # These are the features encoded in the hdf5 'gt' field
        #  A..T: Replacement with A..T, I: Insertion, D: Deletion, .: Unknown;
        #  We will reserve 1 bit for every feature except for 'R' (=Ref base),
        #  which will be ecoded by 0. That leaves us with 14 bits in total,
        #  plus 1 bit for phased or unphased, which will be the low bit,
        #  for easier extraction though shifting.
        snp_features = np.array(['A', 'C', 'G', 'T', 'I', 'D', '.'])
        # We want to encode the left and the right chromosome strand:
        phased_bit = np.uint16(1)
        snp_features_enc = 2**np.arange(1,len(snp_features)+1, dtype=np.uint16)
        
        
        
        # Get dimensions of vcf file
        
        if verbose:
            print("Analyzing input file {}...".format(input_file))
        
        with h5py.File(input_file, "r") as f:
            pos_unenc = np.uint32(f['pos'][:])
            
            max_pos = dataset.get('max_pos', np.max(pos_unenc))
            
            nr_reads = np.int64(pos_unenc.shape[0])
            lengths = np.int32(f['lengths'][:])
            if all_reads:
                max_length = nr_reads
            else:
                max_length = np.max(lengths)
            samples_if = np.array(f['tags'][:], dtype=np.str)
            self.in_memory = in_memory
            if in_memory:
                self.raw_gts = f['gt'][:]
            if verbose:
                print("\t File inforation: {}...".format(f['vcf_info'].value.replace('\n', '\n\t\t')))
                print("\t Max. length: {}...".format(max_length))
                print("\t Max. position: {}...".format(max_pos))
                print("\t Number of reads {}...".format(nr_reads))
                print("\t Number of samples {}...".format(len(samples_if)))
            
        
        if verbose:
            print("Encoding positions...")
        
        dataset['ticker_steps'] = input_encoder.ticker_steps
        input_encoder.calculate_parameters(max_val=max_pos)
        pos_encoded = np.zeros((nr_reads, input_encoder.data_nodes), dtype=np.float32)
        pos_encoded_nonflat = pos_encoded[None,:,:]
        pos_unenc_nonflat = pos_unenc[None,:]
        input_encoder.encode_jit(nr_samples=1, lengths=np.array([nr_reads], dtype=np.uint32), out=pos_encoded_nonflat, positions=pos_unenc_nonflat)
        
        if verbose:
            print("Analyzing target file {}...".format(target_file))
        
        
        # Get dimensions of target file
        
        df_classnames = pa.read_csv(target_file, sep='\t', engine='c', header=None, 
                              lineterminator='\n', dtype=np.object_, 
                              index_col=None, nrows=1)
        if targets is None:
            classes = np.array(df_classnames.iloc[:,1:], dtype=np.str).flatten()
        else:
            classes = targets
        
        # Remove samples without labels for class
        df_targets = pa.read_csv(target_file, sep='\t', engine='c', header=0, 
                              lineterminator='\n', 
                              index_col=0)
        df_targets =  df_targets[classes]      
        samples_tf_full = np.array(df_targets.index, dtype=np.str).flatten()
        df_targets = df_targets.values.astype(np.float32)
        unlabeled_samples = np.all(df_targets == -1, axis=1)
        samples_tf = samples_tf_full
        samples_tf[unlabeled_samples] = np.core.defchararray.add(['no_label_'], samples_tf[unlabeled_samples])
        
        # Check if nr_samples in input and target file matches
        
        # Get samples that exist in both files
        common_samples_mask = np.array([np.char.startswith(np.char.upper(samples_if), s_tf) for s_tf in np.char.upper(samples_tf)], dtype=np.bool)#.reshape(-1,len(samples_if))
        common_samples_inds = np.where(common_samples_mask)
        # Get the positions of the common samples in the input file
        sort_by_if_ind = np.argsort(common_samples_inds[1])
        samples_if_ind = common_samples_inds[1][sort_by_if_ind]
        samples_if_common = samples_if[samples_if_ind]
        # Get the positions of the common samples in the tf file and sort them to the order of the samples in the index file
        samples_tf_ind = common_samples_inds[0][sort_by_if_ind]
        # Get the positions of the skipped samples in the tf file
        common_tf_mask = np.sum(common_samples_mask, axis=1) > 0
        skipped_samples_tf_ind = np.where(~common_tf_mask)[0]
        
        # Remove samples that do not exist in input file and sort according to input file
        self.df_targets = df_targets[samples_tf_ind,:]
        
        # Calculate weighting factors for the classes
        if class_weights:
            pos_classes = self.df_targets == 1
            n_pos_samps = np.sum(pos_classes, axis=0)
            neg_classes = self.df_targets == 0
            n_neg_samps = np.sum(neg_classes, axis=0)
            self.class_weights = n_pos_samps / n_neg_samps
        else:
            self.class_weights = None
        
        if verbose:
            print("\t Number of samples in target file {}...".format(len(samples_tf_full)))
            print("\t Number of classes {}...".format(len(classes)))
            if class_weights:
                print("\t Class weights {}...".format(self.class_weights))
            print("\t Number of labeled samples {}...".format(len(samples_tf)))
            if not (np.all(common_tf_mask) and np.all(np.sum(common_samples_mask, axis=0) > 0)):
                print("Found different samples in input and target file - " +\
                      " reducing samples to {}".format(len(samples_tf_ind)) +\
                      " common labeled entries:\n{} !\n".format(samples_if_common))
        
        
        # If we store the gts in memory, crop to common samples
        if in_memory:
                self.raw_gts = self.raw_gts[samples_if_ind,:]
        
        
        # Precompile reader function via numba
        
        if verbose:
            print("Precompiling reader function...", end="  ")
        
        if all_reads:
            raise AttributeError("Not implemented")
        else:
            @jit(signature_or_function='(uint16[:,:], float32[:,:], int64, int64, int64, uint16[:], float32[:,:,:], uint32[:], uint32[:,:], int64)',
            locals=dict(gts_i=numba.int64, r_i=numba.int64, mb_read_count=numba.int64, code_offset=numba.uint16, code=numba.uint16, tickerstep_bit=numba.uint16, pn_i=numba.int64), nopython=True, nogil=True)
            def decode_gt_fromfile(gts, pos, n_samples, nr_reads, nr_tickersteps,
                             codings, out, pos_unenc, X_unenc, pos_nodes):
                """
                Decode compressed SNV infor from hdf5 file with 
                numba-optimized function (for minibatches read from
                on-the-fly from harddrive)
                
                Notes
                -------
                Not all numba versions can handle range() and slices ([:])
                correctly; that is why they were avoided.
                """
                
                code_offset = 254 # bit mask for masking out phase bit
                
                if nr_tickersteps > 0: # if there are no ticker steps, move position startbit to the left
                    pos_startbit = 13
                else:
                    pos_startbit = 12
                    
                gts_i = 0 # running index for samples
                while(gts_i < n_samples):
                    
                    mb_read_count = 0 #counter for minor alleles found
                    for r_i in range(nr_reads):
                        if (gts[gts_i, r_i] & ~1) != 0: # Check if there is a minor SNP
                            if (gts[gts_i, r_i] & 1) != 0: # phase bit is phased ('/') -> seperate strands
                                # Left chromosome strand:
                                # mask out everything but left chr. gt (=mask out phase bit and right side)
                                code = gts[gts_i, r_i] & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] = 1.
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] = 1.
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] = 1.
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] = 1.
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] = 1.
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] = 1.
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] = 1./6
                                    out[gts_i, mb_read_count, 1] = 1./6
                                    out[gts_i, mb_read_count, 2] = 1./6
                                    out[gts_i, mb_read_count, 3] = 1./6
                                    out[gts_i, mb_read_count, 4] = 1./6
                                    out[gts_i, mb_read_count, 5] = 1./6
                                
                                # Right chromosome strand:
                                # shift right strand to left strand and mask out phase bit
                                code = (gts[gts_i, r_i]>>7) & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 6] = 1.
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 7] = 1.
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 8] = 1.
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 9] = 1.
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 10] = 1.
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 11] = 1.
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 6] = 1./6
                                    out[gts_i, mb_read_count, 7] = 1./6
                                    out[gts_i, mb_read_count, 8] = 1./6
                                    out[gts_i, mb_read_count, 9] = 1./6
                                    out[gts_i, mb_read_count, 10] = 1./6
                                    out[gts_i, mb_read_count, 11] = 1./6
                                
                            else: # phase bit is unphased ('|') -> not clear which strands
                                # Left chromosome strand:
                                # mask out everything but left chr. gt (=mask out phase bit and right side)
                                code = gts[gts_i, r_i] & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] = 0.5
                                    out[gts_i, mb_read_count, 6] = 0.5
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] = 0.5
                                    out[gts_i, mb_read_count, 7] = 0.5
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] = 0.5
                                    out[gts_i, mb_read_count, 8] = 0.5
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] = 0.5
                                    out[gts_i, mb_read_count, 9] = 0.5
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] = 0.5
                                    out[gts_i, mb_read_count, 10] = 0.5
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] = 0.5
                                    out[gts_i, mb_read_count, 11] = 0.5
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] = 1./12
                                    out[gts_i, mb_read_count, 1] = 1./12
                                    out[gts_i, mb_read_count, 2] = 1./12
                                    out[gts_i, mb_read_count, 3] = 1./12
                                    out[gts_i, mb_read_count, 4] = 1./12
                                    out[gts_i, mb_read_count, 5] = 1./12
                                    out[gts_i, mb_read_count, 6] = 1./12
                                    out[gts_i, mb_read_count, 7] = 1./12
                                    out[gts_i, mb_read_count, 8] = 1./12
                                    out[gts_i, mb_read_count, 9] = 1./12
                                    out[gts_i, mb_read_count, 10] = 1./12
                                    out[gts_i, mb_read_count, 11] = 1./12
                                
                                # Right chromosome strand:
                                # shift right strand to left strand and mask out phase bit
                                code = (gts[gts_i, r_i]>>7) & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] += 0.5
                                    out[gts_i, mb_read_count, 6] += 0.5
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] += 0.5
                                    out[gts_i, mb_read_count, 7] += 0.5
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] += 0.5
                                    out[gts_i, mb_read_count, 8] += 0.5
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] += 0.5
                                    out[gts_i, mb_read_count, 9] += 0.5
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] += 0.5
                                    out[gts_i, mb_read_count, 10] += 0.5
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] += 0.5
                                    out[gts_i, mb_read_count, 11] += 0.5
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] += 1./12
                                    out[gts_i, mb_read_count, 1] += 1./12
                                    out[gts_i, mb_read_count, 2] += 1./12
                                    out[gts_i, mb_read_count, 3] += 1./12
                                    out[gts_i, mb_read_count, 4] += 1./12
                                    out[gts_i, mb_read_count, 5] += 1./12
                                    out[gts_i, mb_read_count, 6] += 1./12
                                    out[gts_i, mb_read_count, 7] += 1./12
                                    out[gts_i, mb_read_count, 8] += 1./12
                                    out[gts_i, mb_read_count, 9] += 1./12
                                    out[gts_i, mb_read_count, 10] += 1./12
                                    out[gts_i, mb_read_count, 11] += 1./12
                            
                            # Save encoded position
                            for pn_i in range(pos_nodes):
                                out[gts_i, mb_read_count, pos_startbit+pn_i] = pos[r_i,pn_i]
                            
                            # Save unencoded position
                            X_unenc[gts_i, mb_read_count] = pos_unenc[r_i]
                            
                            # Increase counter (we have found a minor SNP)
                            mb_read_count += 1
                    
                    # now add ticker steps at end (if requested)
                    for r_i in range(nr_tickersteps):
                        out[gts_i, mb_read_count, pos_startbit-1] = 1.
                        X_unenc[gts_i, mb_read_count] = X_unenc[gts_i, mb_read_count-1] + 1
                        mb_read_count += 1
                    
                    # increment sample index   
                    gts_i += 1
            
        
            @jit(signature_or_function='(uint16[:,:], float32[:,:], int64, int64[:], int64, int64, uint16[:], float32[:,:,:], uint32[:], uint32[:,:], int64)',
            locals=dict(gts_i=numba.int64, r_i=numba.int64, mb_read_count=numba.int64, code_offset=numba.uint16, code=numba.uint16, tickerstep_bit=numba.uint16, pn_i=numba.int64), nopython=True, nogil=True)
            def decode_gt_inmem(gts, pos, n_samples, gts_inds, nr_reads, nr_tickersteps,
                             codings, out, pos_unenc, X_unenc, pos_nodes):
                """
                Decode compressed SNV infor from hdf5 file with 
                numba-optimized function (for file stored in memory)
                
                Notes
                -------
                Not all numba versions can handle range() and slices ([:])
                correctly; that is why they were avoided.
                """
                
                code_offset = 254 # bit mask for masking out phase bit
                
                if nr_tickersteps > 0: # if there are no ticker steps, move position startbit to the left
                    pos_startbit = 13
                else:
                    pos_startbit = 12
                    
                gts_i = 0 # running index for samples
                while(gts_i < n_samples):
                    mb_read_count = 0 #counter for minor alleles found
                    for r_i in range(nr_reads):
                        if (gts[gts_inds[gts_i], r_i] & ~1) != 0: # Check if there is a minor SNP
                            if (gts[gts_inds[gts_i], r_i] & 1) != 0: # phase bit is phased ('/') -> seperate strands
                                # Left chromosome strand:
                                # mask out everything but left chr. gt (=mask out phase bit and right side)
                                code = gts[gts_inds[gts_i], r_i] & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] = 1.
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] = 1.
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] = 1.
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] = 1.
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] = 1.
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] = 1.
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] = 1./6
                                    out[gts_i, mb_read_count, 1] = 1./6
                                    out[gts_i, mb_read_count, 2] = 1./6
                                    out[gts_i, mb_read_count, 3] = 1./6
                                    out[gts_i, mb_read_count, 4] = 1./6
                                    out[gts_i, mb_read_count, 5] = 1./6
                                
                                # Right chromosome strand:
                                # shift right strand to left strand and mask out phase bit
                                code = (gts[gts_inds[gts_i], r_i]>>7) & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 6] = 1.
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 7] = 1.
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 8] = 1.
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 9] = 1.
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 10] = 1.
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 11] = 1.
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 6] = 1./6
                                    out[gts_i, mb_read_count, 7] = 1./6
                                    out[gts_i, mb_read_count, 8] = 1./6
                                    out[gts_i, mb_read_count, 9] = 1./6
                                    out[gts_i, mb_read_count, 10] = 1./6
                                    out[gts_i, mb_read_count, 11] = 1./6
                                
                            else: # phase bit is unphased ('|') -> not clear which strands
                                # Left chromosome strand:
                                # mask out everything but left chr. gt (=mask out phase bit and right side)
                                code = gts[gts_inds[gts_i], r_i] & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] = 0.5
                                    out[gts_i, mb_read_count, 6] = 0.5
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] = 0.5
                                    out[gts_i, mb_read_count, 7] = 0.5
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] = 0.5
                                    out[gts_i, mb_read_count, 8] = 0.5
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] = 0.5
                                    out[gts_i, mb_read_count, 9] = 0.5
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] = 0.5
                                    out[gts_i, mb_read_count, 10] = 0.5
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] = 0.5
                                    out[gts_i, mb_read_count, 11] = 0.5
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] = 1./12
                                    out[gts_i, mb_read_count, 1] = 1./12
                                    out[gts_i, mb_read_count, 2] = 1./12
                                    out[gts_i, mb_read_count, 3] = 1./12
                                    out[gts_i, mb_read_count, 4] = 1./12
                                    out[gts_i, mb_read_count, 5] = 1./12
                                    out[gts_i, mb_read_count, 6] = 1./12
                                    out[gts_i, mb_read_count, 7] = 1./12
                                    out[gts_i, mb_read_count, 8] = 1./12
                                    out[gts_i, mb_read_count, 9] = 1./12
                                    out[gts_i, mb_read_count, 10] = 1./12
                                    out[gts_i, mb_read_count, 11] = 1./12
                                
                                # Right chromosome strand:
                                # shift right strand to left strand and mask out phase bit
                                code = (gts[gts_inds[gts_i], r_i]>>7) & code_offset
                                if code == 0: # no minor SNP on this strand
                                    pass
                                elif code == codings[0]: #'A'
                                    out[gts_i, mb_read_count, 0] += 0.5
                                    out[gts_i, mb_read_count, 6] += 0.5
                                elif code == codings[1]: #'C'
                                    out[gts_i, mb_read_count, 1] += 0.5
                                    out[gts_i, mb_read_count, 7] += 0.5
                                elif code == codings[2]: #'G'
                                    out[gts_i, mb_read_count, 2] += 0.5
                                    out[gts_i, mb_read_count, 8] += 0.5
                                elif code == codings[3]: #'T'
                                    out[gts_i, mb_read_count, 3] += 0.5
                                    out[gts_i, mb_read_count, 9] += 0.5
                                elif code == codings[4]: #'I'
                                    out[gts_i, mb_read_count, 4] += 0.5
                                    out[gts_i, mb_read_count, 10] += 0.5
                                elif code == codings[5]: #'D'
                                    out[gts_i, mb_read_count, 5] += 0.5
                                    out[gts_i, mb_read_count, 11] += 0.5
                                elif code == codings[6]: #'.'
                                    out[gts_i, mb_read_count, 0] += 1./12
                                    out[gts_i, mb_read_count, 1] += 1./12
                                    out[gts_i, mb_read_count, 2] += 1./12
                                    out[gts_i, mb_read_count, 3] += 1./12
                                    out[gts_i, mb_read_count, 4] += 1./12
                                    out[gts_i, mb_read_count, 5] += 1./12
                                    out[gts_i, mb_read_count, 6] += 1./12
                                    out[gts_i, mb_read_count, 7] += 1./12
                                    out[gts_i, mb_read_count, 8] += 1./12
                                    out[gts_i, mb_read_count, 9] += 1./12
                                    out[gts_i, mb_read_count, 10] += 1./12
                                    out[gts_i, mb_read_count, 11] += 1./12
                            
                            # Save encoded position
                            for pn_i in range(pos_nodes):
                                out[gts_i, mb_read_count, pos_startbit+pn_i] = pos[r_i,pn_i]
                            
                            # Save unencoded position
                            X_unenc[gts_i, mb_read_count] = pos_unenc[r_i]
                            
                            # Increase counter (we have found a minor SNP)
                            mb_read_count += 1
                    
                    # now add ticker steps at end (if requested)
                    for r_i in range(nr_tickersteps):
                        out[gts_i, mb_read_count, pos_startbit-1] = 1.
                        X_unenc[gts_i, mb_read_count] = X_unenc[gts_i, mb_read_count-1] + 1
                        mb_read_count += 1
                    
                    # increment sample index   
                    gts_i += 1
                    
        if verbose:
            print("done!")
        
        # Save information about dataset
        
        dataset['max_pos'] = max_pos
        dataset['ticker_steps'] = ticker_steps
        dataset['input_encoder'] = input_encoder
        # input features/nodes: SNP type + ticker steps + position
        dataset['n_features'] = int((len(snp_features_enc)-1 + len(snp_features_enc)-1) + (dataset['ticker_steps']>0) + input_encoder.data_nodes) #-1 because of '.'
        dataset['n_classes'] = len(classes)
        dataset['max_len'] = int(max_length + dataset['ticker_steps'])
        dataset['batch_size'] = int(batch_size)
        
        dataset['batch_size_X'] = (dataset['batch_size'], dataset['max_len'], dataset['n_features'])
        dataset['batch_size_y'] = (dataset['batch_size'], dataset['n_classes'])
        dataset['batch_size_mask'] = (dataset['batch_size'], dataset['max_len'])
        # batch_size_length = position of final timestep
        dataset['batch_size_last_ind'] = (dataset['batch_size'])
        # batch_size_weights = weights for predictions (good for masking out)
        dataset['batch_size_weights'] = (dataset['batch_size'], dataset['n_classes'])
        
        
        self.pos_encoded = pos_encoded
        self.nr_reads = nr_reads
        self.max_length = np.uint32(max_length)
        self.sample_names = samples_if_common
        self.skipped_samples_tf_ind = skipped_samples_tf_ind
        self.input_file = input_file
        self.target_file = target_file
        self.classes = classes
        self.phased_bit = phased_bit
        self.snp_features_enc = np.uint16(snp_features_enc)
        self.snp_features = snp_features
        self.pos_unenc = pos_unenc
        self.pos_nodes = input_encoder.data_nodes
        
        if in_memory:
            # mask for matrix with common entries - covers common samples
            self.samples_if_mask = np.zeros_like(samples_if_common, dtype=np.bool)
            # same for lenghts of samples
            self.lengths = lengths[samples_if_ind]
        else:
            # need to read from hdf5 file - mask covers all samples
            self.samples_if_mask = np.zeros_like(samples_if, dtype=np.bool)
            # same for lenghts of samples
            self.lengths = lengths
        
        self.samples_if_ind = samples_if_ind
        self.samples_tf_ind = samples_tf_ind
        
        self.dataset = dataset
        if in_memory:
            self.decode_gt_inmem = decode_gt_inmem
        else:
            self.decode_gt_fromfile = decode_gt_fromfile
    
    def load_data_mb(self, sample_inds, verbose=True):
        """
        Load a minibatch
        
        Parameters
        -------
        sample_inds : list or np.array of integers
            Indices of samples to read (indices refer to the position in data
            file), i.e. sample_inds in train, valid, or test set.
        Returns
        -------
        Minibatch as dictionary
        """
        dataset = self.dataset
        nr_reads = self.nr_reads
        sample_names = self.sample_names
        snp_features_enc = self.snp_features_enc
        input_file = self.input_file
        sample_inds = np.array(sample_inds, dtype=np.int64)
        sample_inds.sort()
        pos_encoded = self.pos_encoded
        pos_unenc = self.pos_unenc
        pos_nodes = self.pos_nodes
        in_memory = self.in_memory
        lengths = self.lengths
        
        samples_if_mask = self.samples_if_mask 
        samples_if_ind = self.samples_if_ind
        
        if verbose:
            t0 = time.time()
        # Create empty minibatch
        minibatch = dict()
        # input activations will be declared later
        minibatch['X'] = np.zeros(dataset['batch_size_X'], dtype=np.float32)
        # mask out targets with -1
        minibatch['y'] = np.zeros(dataset['batch_size_y'], dtype=np.float32) - 1
        # input mask for variable input lengths
        minibatch['mask'] = np.zeros(dataset['batch_size_mask'], dtype=np.int8)
        # weights for loss-weighting -> will mask out elements=0
        minibatch['weights'] = np.empty(dataset['batch_size_weights'], dtype=np.float32)
        # names of the samples (optional)
        minibatch['names'] = np.zeros(dataset['batch_size'], dtype=sample_names.dtype)
        # store unencoded positions for plotting (expensive but meh...)
        minibatch['X_unenc'] = np.zeros((len(sample_inds), dataset['batch_size_X'][1]), dtype=np.uint32)
        # last element to be onsidered in sequences (incl. tickersteps)
        minibatch['last_ind'] = np.zeros(dataset['batch_size_last_ind'], dtype=np.int32)
        
        # The existing samples and their indices/mask in the minibatch (h5py only supports boolean pointselection)
        samples_if_mask[:] = 0
        if in_memory:
            # mask only has the sample_names entries and the sample_inds correspond to the index of sample_names
            samples_if_mask[sample_inds] = 1
        else:
            # mask has all (sample_if) entries and the sample_inds correspond to the index of sample_names
            samples_if_mask[samples_if_ind[sample_inds]] = 1
        
        # Set names of samples
        minibatch['names'][:len(sample_inds)] = sample_names[sample_inds]
        
        
        if verbose:
            t1 = time.time()
            
        # Set lengths last_ind
            
        minibatch['last_ind'][:len(sample_inds)] = lengths[samples_if_mask] + dataset['ticker_steps'] - 1
        # Create mask for timesteps/sequence positions
        for s_i in range(dataset['batch_size']):
            minibatch['mask'][s_i,:minibatch['last_ind'][s_i]] = 1
        
        # Set input activations X 
        if verbose:
            t2 = time.time()
        
        if in_memory:
            #gts = self.raw_gts[samples_if_mask,:]
            
            self.decode_gt_inmem(gts=self.raw_gts,
                 pos=pos_encoded,
                 n_samples=np.int64(len(sample_inds)),
                 #sample_inds=np.uint32(samples_inds_mb),
                 gts_inds=sample_inds,
                 nr_reads=np.int64(nr_reads),
                 nr_tickersteps=dataset['ticker_steps'],
                 codings=snp_features_enc,
                 out=minibatch['X'],
                 pos_unenc=pos_unenc,
                 X_unenc=minibatch['X_unenc'],
                 pos_nodes=pos_nodes)
        else:
            with h5py.File(input_file, "r") as f:
                gts = f['gt'][samples_if_mask,:]
            
            self.decode_gt_fromfile(gts=gts,
                 pos=pos_encoded,
                 n_samples=np.int64(len(sample_inds)),
                 #sample_inds=np.uint32(samples_inds_mb),
                 nr_reads=np.int64(nr_reads),
                 nr_tickersteps=dataset['ticker_steps'],
                 codings=snp_features_enc,
                 out=minibatch['X'],
                 pos_unenc=pos_unenc,
                 X_unenc=minibatch['X_unenc'],
                 pos_nodes=pos_nodes)
        
        if verbose:
            t3 = time.time()
        
        # Set targets y
        
        #(dataset['batch_size'], dataset['n_classes'])
        minibatch['y'][:len(sample_inds),:] = self.df_targets[sample_inds,:]
        
        if verbose:
            t4 = time.time()
        
        # Set weights/targetmask
        minibatch['weights'][:] = minibatch['y'] != -1 #-1 = masked out target
        if self.class_weights is not None:
            for ci, cw in enumerate(self.class_weights):
                if cw > 1:
                    minibatch['weights'][minibatch['y'][:,ci]==1, ci] /= cw
                else:
                    minibatch['weights'][minibatch['y'][:,ci]==0, ci] *= cw
        
        # Clean up y's masked out elements
        
        minibatch['y'][minibatch['y'] == -1] = 0.5
        if verbose:
            t5 = time.time()
        
        if verbose:
            print("{} {} {} {} {} {}".format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0))
        
        return minibatch
    

if __name__ == '__main__':
    pass