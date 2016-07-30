# -*- coding: utf-8 -*-
"""vcf_to_hdf5_tools.py: Convert VCF files to HDF5 files


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Go to the bottom of the file for an example script of how to convert all
VCF files in a directory to HDF5 automatically.

VCF_to_H5(): Class for conversion
    convert_to_h5_phased(): Start conversion
convert_dir(): Function for converting all files in a directory in parallel
    threads

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import sys, os
import numpy as np
import gzip
import time
import pandas as pa
import h5py
import re
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from michaels_modules.net_utils.input_encoders_numba import InputEncoderTriangles, InputEncoderBinary

def replace_dots(x):
    try:
        return int(x)
    except ValueError:
        return -1

class VCF_to_H5(object):
    def __init__(self, vcf_filename, dots_in_samples=True, verbose=True, 
                 chunksize=10000):
        """Initialize VCF file for conversion to HDF5
        
        Parameters
        -------
        vcf_filename : string
            Name of VCF file to convert
        dots_in_samples : bool
            Can dots be in GT columns too? (e.g. './0' in GT column of VCF 
            file - this is not a standard VCF then); set to True if unsure;
        chunksize : int
            Files are converted in chunks. chunksize sets the number of lines in
            the VCF file to load into memory at once. Decrease if RAM too small
        Returns
        -------
        Nothing but provides function to start conversion
        """
        # These are the features we want to encode for:
        #  A..T: Replacement with A..T, I: Insertion, D: Deletion, .: Unknown;
        #  We will reserve 1 bit for every feature except for 'R' (=Ref base),
        #  which will be ecoded by 0. That leaves us with 14 bits in total,
        #  plus 1 bit for phased or unphased, which will be the low bit,
        #  for easier extraction though shifting.
        snp_features = np.array(['A', 'C', 'G', 'T', 'I', 'D', '.'])
        # We want to encode the left and the right chromosome strand:
        phased_bit = np.uint16(1)
        snp_features_l = dict(zip(snp_features, 2**np.arange(1,len(snp_features)+1, dtype=np.uint16))) #1 -> no phased bit offset
        snp_features_r = 7# nr bits to shift dict(zip(snp_features, 2**np.arange(len(snp_features)+1, len(snp_features)*2+1, dtype=np.uint16)))
        # Note: the high bit will remain unused, but gzip compression with the
        #  right flag shoult be able to take advantage of that
        # view with np.binary_repr(a, width=16)
        
        
        ## Get dimensions of vcf file
        
        if verbose:
            print("Analyzing input file {}:".format(vcf_filename))
            print("\t...data positions".format(vcf_filename))
        
        vcf_data_pos = dict(data_start=-1, pos=-1, ref=-1, qual=-1, alt=-1, gt_start=-1, gt_stop=-1)
        
        with gzip.open(vcf_filename, 'rt') as gz:
            data_reached = False
            for (l_i, line) in enumerate(gz):
                if line.startswith('#CHROM'):
                    data_reached = True
                    vcf_data_pos['data_start'] = l_i + 1
                    tag_line = np.array(line.strip().split('\t'), dtype=(np.str, 4))
                    continue
                if data_reached:
                    np_line = np.array(line.strip().split('\t'), dtype=str)
                    if l_i == vcf_data_pos['data_start']:
                        for (c_i,col) in enumerate(np_line):
                            if col.startswith('GT'):
                                vcf_data_pos['gt_start'] = c_i+1
                                vcf_data_pos['gt_stop'] = len(np_line)
                                break
                        vcf_data_pos['ref'] = np.where(tag_line == 'REF')[0][0]
                        vcf_data_pos['alt'] = np.where(tag_line == 'ALT')[0][0]
                        vcf_data_pos['qual'] = np.where(tag_line == 'QUAL')[0][0]
                        vcf_data_pos['pos'] = np.where(tag_line == 'POS')[0][0]
                        nr_samples = vcf_data_pos['gt_stop']-vcf_data_pos['gt_start']
                        break
        
        
        if verbose:
            print("\t...lengths of samples", end='\r')
        
        
        # Get lenghts of sequence per sample
        
        #lengths = np.zeros(nr_samples, dtype=np.uint32)
        
        # Set params and create file iterator
        # indices of sample columns
        usecols = np.arange(vcf_data_pos['gt_start'], vcf_data_pos['gt_stop'])
        skiprows = vcf_data_pos['data_start']-1
        header = 0#vcf_data_pos['data_start']
        chunksize = chunksize
        cmpr = [None, 'gzip'][vcf_filename.endswith('.gz')]
        
        lengths = np.zeros(nr_samples, dtype=np.uint32)
        no_snp = np.array(['0/0'],  dtype=(np.str_, 3))
        np_line = np.zeros((chunksize, nr_samples), dtype=(np.str_, 3))
        
        with open(vcf_filename, 'rb') as vf:
            df_iter = pa.read_csv(vf, sep='\t',
                                  engine='c', lineterminator='\n',
                                  skiprows=skiprows, comment=None, 
                                  header=header, index_col=None, nrows=None, 
                                  iterator=True, chunksize=chunksize, 
                                  usecols=usecols, compression=cmpr)
            
            nr_reads = 0
            for df_line_i, df_line in enumerate(df_iter):
                if verbose:
                    print("\t...lengths of samples (chunk {}, size {})".format(df_line_i, df_line.shape), end='\r')
                    sys.stdout.flush()
                nr_reads += df_line.shape[0]
                np_line[:df_line.shape[0],:] = df_line
                lengths += np.uint32(np.sum((np_line[:df_line.shape[0],:] != no_snp), axis=0))
            else:
                samples_vcf = np.array(df_line.columns, dtype=np.str)
        
        del df_iter
        del np_line
        
        if verbose:
            print("\t...lengths of samples")
            sys.stdout.flush()
        max_length = np.max(lengths)
        
        if verbose:
            print("\t...highest position (on genome)".format(vcf_filename))
            sys.stdout.flush()
            
        with open(vcf_filename, 'rb') as vf:
            # Get highest position (on genome)
            
            df_full = pa.read_csv(vf, sep='\t',
                                  engine='c', lineterminator='\n', 
                                  skiprows=skiprows, comment=None, 
                                  dtype=np.uint32, header=header, index_col=None, 
                                  nrows=None, iterator=False, compression=cmpr, 
                                  usecols=[vcf_data_pos['pos']])
        max_pos = np.max(df_full.iloc[:,0])
        
        if verbose:
            print("Counted {} reads in {} samples. Maximal position: {}.".format(nr_reads, nr_samples, max_pos))
            sys.stdout.flush()
        self.lengths = lengths
        self.nr_reads = nr_reads
        self.max_length = max_length
        self.vcf_data_pos = vcf_data_pos
        self.samples = samples_vcf
        self.vcf_filename = vcf_filename
        self.snp_features = snp_features
        self.vcf_compression = cmpr
        self.dots_in_samples = dots_in_samples
        self.snp_features_l = snp_features_l
        self.snp_features_r = snp_features_r
        self.phased_bit = phased_bit
        self.max_pos = max_pos
        #if phasetype == 'single':
        #    pos_ind = len(self.snp_encoding_l)
    
    
    def convert_to_h5_phased(self, output_file, h5comp_dict=None, 
                             force_phased=False, verbose=True, 
                             sample_chunksize=1):
        """Start conversion of VCF to HDF5 file
        
        Parameters
        -------
        output_file : string
            Name of HDF5 file to create
        h5comp_dict : dict
            Compression specifications for HDF5 file
        force_phased : bool
            Force all VCF GT fields to be treated as phased?
        sample_chunksize : int
            number of samples to convert at once. Similar to chunksize, but
            regarding the samples (columns) in the VCF file. Prefer relatively
            high values here instead of chunksize, for faster conversion.
            
        Returns
        -------
        creates HDF5 file output_file
        """
        
        vcf_filename = self.vcf_filename
        vcf_data_pos = self.vcf_data_pos
        vcf_cmpr = self.vcf_compression
        nr_reads = self.nr_reads
        sample_tags = self.samples
        lengths = self.lengths
        dots_in_samples = self.dots_in_samples
        max_pos = self.max_pos
        
        snp_features_l = self.snp_features_l
        snp_features_r = self.snp_features_r
        
        sample_inds = np.arange(len(sample_tags), dtype=np.int32)
        
        nr_samples = len(sample_inds)
        
        
        ## Initialize position encoders
        
        tria_encoder = InputEncoderTriangles(ticker_steps=0, min_dif=1e-5, add_binary_encoding=True)
        tria_encoder.calculate_parameters(max_val=max_pos)
        
        bina_encoder = InputEncoderBinary(ticker_steps=0)
        bina_encoder.calculate_parameters(max_val=max_pos)
        
        
        ## Create empty hdf5 file
        
        if h5comp_dict == None:
            h5comp_dict = dict(compression='gzip', compression_opts=9, shuffle=True)
        
        with h5py.File(output_file+".hdf5", "w") as f:
            f.create_dataset('vcf_info', data="VCF filename: {}\nHDF5 filename: {}\nPhase type: phased\nPhase type forced: {}".format(vcf_filename, output_file, force_phased),
                             dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('tags', data=[t.encode('utf8') 
                             for t in sample_tags], **h5comp_dict) #mind utf8
            f.create_dataset('lengths', data=lengths, 
                             dtype=np.uint32, scaleoffset=0, fillvalue=0,**h5comp_dict)
            f.create_dataset('gt', shape=(nr_samples, nr_reads), 
                             dtype=np.uint16, scaleoffset=0, fillvalue=0,**h5comp_dict)           
            f.create_dataset('pos', shape=(nr_reads,), 
                             dtype=np.uint32, scaleoffset=0, fillvalue=0,**h5comp_dict)
            # minibatch shape will be (dataset['batch_size'], 
            #     dataset['max_len'], dataset['n_features'])
        
        
        
        ## Load sample by sample from dataset and write it to hdf5 file
        
        # Get POS, REF, and ALT fields of all reads
        
        # We need position, ref- and alt bases
        usecols = [vcf_data_pos['pos'], vcf_data_pos['ref'],
                             vcf_data_pos['alt']]
        # Get a complete samples at once
        iterator = False
        # Skip the header
        skiprows = vcf_data_pos['data_start']
        # Datatypes are ignored for converters
        dtypes = {vcf_data_pos['pos']: np.uint32}
        # Create converter to spilt ALT field t ','s
        converters = {vcf_data_pos['alt']: lambda x : x.split(',')}
        # Read POS, REF, and ALT from file
        if verbose:
            print("Reading from file {}:".format(vcf_filename))
            print("\t...POS, REF, ALT")
            sys.stdout.flush()
        
        with open(vcf_filename, 'rb') as vf:
            df = pa.read_csv(vf, sep='\t', compression=vcf_cmpr, 
                             engine='c', lineterminator='\n', dtype=dtypes, 
                             header=None, index_col=None, iterator=iterator, 
                             converters=converters, skiprows=skiprows,
                             usecols=usecols)
        
        # Write positions directly to hdf5 file
        with h5py.File(output_file+".hdf5", "a") as hf:
            hf['pos'][:,] = np.uint32(df[vcf_data_pos['pos']])
        
        # Store ref and alt for later
        df_ref = np.uint8(df[vcf_data_pos['ref']].apply(np.char.str_len))
        df_alt = np.array(df[vcf_data_pos['alt']], dtype=np.object_)
        del df
        
        if verbose:
            print("\t...parsing")
            sys.stdout.flush()
        # Change df_alt into key for look-up-dict for encoding of bases later
        # add a 'R' key to dict, coding for ref (e.g 0 in. '0/1') as 0 (is ignored)
        # connect to snp_features_l - for snp_features_r just shift results
        if dots_in_samples: #dots are in sample columns, so add them to alt
            def connect_lookup(l_ref, alt):
                if l_ref == 1: #no deletions possible, because only ref element (see replace_dots())
                    # if key can't be found, it's an insertion
                    return np.array([0] + \
                            [snp_features_l.get(a, snp_features_l['I']) 
                            for a in alt] + [snp_features_l['.']], dtype=np.uint16)
                else:
                    return np.array([0] + \
                            [snp_features_l['.'] if len(a) == l_ref else 
                            snp_features_l['D'] if len(a) < l_ref else 
                            snp_features_l['I'] for a in alt] + \
                            [snp_features_l['.']], dtype=np.uint16)
        else:
            def connect_lookup(l_ref, alt):
                if l_ref == 1: #no deletions possible, because only ref element
                    # if key can't be found, it's an insertion
                    return np.array([0] + \
                            [snp_features_l.get(a, snp_features_l['I']) 
                            for a in alt], dtype=np.uint16)
                else:
                    return np.array([0] + \
                            [snp_features_l['.'] if len(a) == l_ref else 
                            snp_features_l['D'] if len(a) < l_ref else 
                            snp_features_l['I'] for a in alt], dtype=np.uint16)
        
        df_alt = [connect_lookup(df_ref[i], df_alt[i]) for i in range(nr_reads)]
        
        
        # Get GT fields for all samples and write map to h5py
        
        if verbose:
            print("\t...GT")
            sys.stdout.flush()
        
        def parse_gts(alt_enc, gts):
            """sample_gt[0,1] holds 2 bytes with the low bits as index for
            SNV type and high bit for phased bit and major/minor bit
            
            Note: Make sure the dots_in_samples flag is correctly set or
            at True!"""
            if gts[0:3:2]!='00':
                s = re.split('([/|:])', gts, 2) #re.split pays off at >2 delimiters
                return  np.bitwise_or(\
                            np.bitwise_or(alt_enc[replace_dots(s[0])], 
                                          gts[len(s[1])]=='/'),
                            np.left_shift(alt_enc[replace_dots(s[2])], snp_features_r))
            else:
                return gts[2]=='/' # first bit encodes phase, rest is 0
        
        
        # Preallocate numpy array as template for final data
        sample_gts = np.empty((nr_reads), dtype=np.uint16)
        full_range = range(nr_reads)
        
        # And a numpy container for the read 
        df = np.zeros((nr_reads, sample_chunksize), dtype=np.object_)
        
        # Now we want the GT columns, one chunk of samples at a time
        for s_i, s_ind in enumerate(sample_inds):
            if verbose:
                print("\t...sample {}".format(s_ind))
                sys.stdout.flush()
            
            if verbose:
                t1 = time.time()
            
            if (s_i % sample_chunksize) == 0:
                if s_i+sample_chunksize <= len(sample_inds):
                    chunk_end = sample_chunksize
                else:
                    chunk_end = len(sample_inds) % sample_chunksize
                
                # Read current sample chunk only
                usecols = sample_inds[s_i:s_i+chunk_end] + vcf_data_pos['gt_start']
                # Datatypes are ignored for converters
                dtypes = None
                # Create converter to spilt ALT field t ','s
                converters = None#{usecols[0]: extract_gts}
                # Read POS, REF, and ALT from file
                with open(vcf_filename, 'rb') as vf:
                    df[:,:chunk_end] = pa.read_csv(vf, sep='\t', compression=vcf_cmpr, 
                                     engine='c', lineterminator='\n', dtype=dtypes, 
                                     header=None, index_col=None, iterator=iterator, 
                                     converters=converters, skiprows=skiprows,
                                     usecols=usecols)[usecols].values
                
                
            if verbose:
                t2 = time.time()
            
            sample_gts[:] = [parse_gts(df_alt[r], df[r,s_i%sample_chunksize]) for r in full_range]
            
            # now write to hdf5 file
            if verbose:
                t3=time.time()
            
            with h5py.File(output_file+".hdf5", "a") as hf:
                hf['gt'][s_ind, :] = sample_gts
            
            if verbose:
                t4=time.time()
                print("\t\tRead: {}, Parse: {}, Write: {}, Total: {} sec".format(t2-t1,t4-t2,t4-t3,t4-t1))
                sys.stdout.flush()
        
        print("Created file {}".format(output_file+'.hdf5'))
        sys.stdout.flush()

def convert_dir(dirpath, regex=r'(.*)\.(vcf.gz)$', 
                h5comp_dict=dict(compression='gzip', compression_opts=9), 
                multicore=False, verbose=True, verbose_subprocesses=False, 
                chunksize=int(5*1e4), sample_chunksize=100):
    """Convert all VCF files in a directory in parallel threads
    
    Parameters
    -------
    dirpath : string
        Path to VCF files
    regex : string
        Regec for file extension of vcf files e.g. '(.*)\.(vcf.gz)$' or
        '(.*)\.(vcf)$'
    h5comp_dict : dict
        Compression specifications for HDF5 file
    multicore : int
        0: No multiprocessing
        >0: Number of parallel threads to spawn
    chunksize : int
        Files are converted in chunks. chunksize sets the number of lines in
        the VCF file to load into memory at once. Decrease if RAM too small
    sample_chunksize : int
        number of samples to convert at once. Similar to chunksize, but
        regarding the samples (columns) in the VCF file. Prefer relatively
        high values here instead of chunksize, for faster conversion.
        
    Returns
    -------
    creates HDF5 files in dirpath
    """
    
    datafiles = [os.path.join(dirpath,f) for f in os.listdir(dirpath) if re.search(regex, f)]
    if multicore:
        from multiprocessing import Pool
        converters = list()
        for df_i, datafile in enumerate(datafiles):
            if verbose:
                print("Getting infos for {} ({})...".format(datafile, df_i))
            converters.append(VCF_to_H5(datafile, chunksize=chunksize, verbose=verbose))
            if ((df_i+1) % multicore) == 0:
                pool = Pool(processes=multicore)
                for converter in converters:
                    if verbose:
                        print("Starting parsing process for {}...".format(converter.vcf_filename))
                    pool.apply_async(converter.convert_to_h5_phased,[converter.vcf_filename, h5comp_dict, False, verbose_subprocesses, sample_chunksize])
                pool.close()
                pool.join()
                del converters
                del pool
                converters = list()
        else:
            # parse the remaining files
            pool = Pool(processes=multicore)
            for converter in converters:
                if verbose:
                    print("Starting parsing process for {}...".format(converter.vcf_filename))
                pool.apply_async(converter.convert_to_h5_phased,[converter.vcf_filename, h5comp_dict, False, verbose_subprocesses, sample_chunksize])
            pool.close()
            pool.join()
            del converters
            del pool
        
    else:
        for datafile in datafiles:
            converter = VCF_to_H5(datafile, chunksize=chunksize, verbose=verbose)
            vcf_file = re.search(regex, datafile).groups()[0]
            if verbose:
                print("Starting parsing for {}...".format(converter.vcf_filename))
            converter.convert_to_h5_phased(vcf_file, h5comp_dict=h5comp_dict, sample_chunksize=sample_chunksize)
    
    print("Done!")
        

if __name__ == '__main__':
    t1 = time.time()
    convert_dir('path/to/vcf_files/', multicore=0, chunksize=int(5*1e4), sample_chunksize=100)
    print("Done! Duration: {} sec".format(time.time()-t1))
    