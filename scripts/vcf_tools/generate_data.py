# -*- coding: utf-8 -*-
"""generate_data.py: Generate artificial SNV datasets as VCF files


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Script for generation of artificial SNV datasets as VCF files and the 
corresponding target file as CSV file

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-29  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""


############################################################################
#  Imports                                                                 #
############################################################################

import os, sys

import numpy as np
from itertools import chain
from collections import OrderedDict
from scipy.stats.distributions import norm as scipy_normal
from matplotlib import pyplot as pl
import pprint
import gzip

sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from michaels_modules.utility.external_sources import make_sure_path_exists


############################################################################
#  Parameters                                                              #
############################################################################

verbose = True
# number of threads to spawn
multicore = 3
# large vcf file -> write line by line in junks
large_vcf_file = False

# number of samples per positive and negative class
n_pos_seqs = 100
n_neg_seqs = 100

# Statistics from bacteria dataset:
#  % of genome are minor SNVs in samples:
#   max. 0.0015919% (3rd max: 0.0010707%), min.: 4.9101e-05%
#  Sequence lenght: 6 537 437
#  Max. nr. of SNPs per samlpe: 10407 (3rd max: ~7000) (49024)
#  Min. nr. of SNPs per samlpe: 321 (16228)
#  Avg. nr. of SNPs per samlpe: 3113 (32200)
#  Nr of samples: 151
#  bacteria_nr_snps = np.array([5380,5380,1975,1975,3214,3214,4551,4551,2896,2896,4508,4508,3316,3316,3239,3239,5180,5180,373,373,2029,2029,2749,2749,2440,2440,4978,4978,2739,2739,4546,4546,3572,3572,4499,4499,5322,5322,1770,1770,6069,6069,3368,3368,4553,4553,1933,1933,3950,3950,3611,3611,6169,6169,6321,6321,6542,6542,2918,2918,5602,5602,1516,1516,7662,7662,10407,10407,3088,3088,4350,4350,2619,2619,1452,1452,2910,2910,3706,3706,490,490,1112,1112,1457,1457,2415,2415,2699,2699,1371,1371,3773,3773,1877,1877,1355,1355,964,964,4410,4410,629,629,2624,2624,1229,1229,3137,3137,477,477,3310,3310,684,684,1782,1782,2182,2182,1926,1926,875,875,1148,1148,1897,1897,1637,1637,1336,1336,3231,3231,4109,4109,4625,4625,818,818,1906,1906,2954,2954,1504,1504,3404,3404,1737,1737,4018,4018,3924,3924,6967,6967,3932,3932,2181,2181,3319,3319,3550,3550,4096,4096,4231,4231,3044,3044,5200,5200,1842,1842,3462,3462,4908,4908,3583,3583,1571,1571,3376,3376,2592,2592,5331,5331,1897,1897,4229,4229,4270,4270,2861,2861,1584,1584,2795,2795,4207,4207,2488,2488,3024,3024,2406,2406,1556,1556,2763,2763,2869,2869,1948,1948,3394,3394,1319,1319,4677,4677,5052,5052,3428,3428,3458,3458,3213,3213,2273,2273,3763,3763,2259,2259,1897,1897,3068,3068,3187,3187,3693,3693,3479,3479,321,321,3184,3184,2683,2683,1856,1856,2810,2810,4739,4739,4461,4461,3798,3798,4237,4237,2835,2835,4614,4614,1608,1608,3826,3826,3693,3693,2195,2195,3974,3974,3525,3525,1723,1723,2169,2169,2383,2383,3266,3266,3966,3966,1838,1838,3800,3800,3906,3906,2807,2807,2188,2188,1517,1517])[::2]
#  pl.hist(bacteria_nr_snps, bins=50)


## Original data statistics

original_data = dict(seq_len=6537437, min_len=16228, max_len=49024, 
                     mean_len=32200)

original_data['min_rat'] = (float(original_data['min_len']) / 
    original_data['seq_len'])
original_data['max_rat'] = (float(original_data['max_len']) / 
    original_data['seq_len'])
original_data['mean_rat'] = (float(original_data['mean_len']) / 
    original_data['seq_len'])
print("Original data statistics:\n\t{}".format(original_data))


############################################################################
#  Create artificial datasets                                              #
############################################################################

for dataset in range(1,5):
    
    dirname = "SNV_Artificial_{}".format(int(dataset))
    make_sure_path_exists(dirname)
    
    if dataset == 1:
        # Dataset 1
        # seqence length of "genome"
        seq_len = 500000
        # percentages of minor SNVs per sample
        min_perc = 0.005
        max_perc = 0.09
        
    elif dataset == 2:
        # Dataset 2
        seq_len = 500000
        min_perc = 0.25
        max_perc = 0.75
        
    elif dataset == 3:
        # Dataset 3
        seq_len = 6500000
        min_perc = 0.005
        max_perc = 0.09
        
    elif dataset == 4:
        # Dataset 4
        seq_len = 6500000
        min_perc = 0.25
        max_perc = 0.75
    
    
    mean_perc = (max_perc - min_perc) / 2 + min_perc
    
    min_len = int(seq_len * min_perc / 100)
    max_len = int(seq_len * max_perc / 100)
    mean_len = int(seq_len * mean_perc/ 100)
    
    # Probabilities of SNP positions by normal distribution
    distribution = 'normal'
    loc = mean_len
    scale = (mean_len-min_len + max_len-mean_len) / 2
    
    distr = scipy_normal(loc=loc, scale=scale)
    x = np.arange(min_len, max_len)
    probs = distr.pdf(x)
    # sum probs to 1
    probs /= probs.sum()
    pl.figure()
    pl.plot(x, probs)
    pl.title('Numbers of SNPs per sample')
    
    artificial_data = OrderedDict()
    for i in ('seq_len', 'max_len', 'min_len', 'mean_len', 'distribution', 
    'loc', 'scale'):
        artificial_data[i] = locals()[i]
    print("Artificial data statistics:\n{}".format(pprint.pformat(artificial_data)))
    
    snps_per_sample = np.random.choice(a=x, p=probs, size=1e4)
    pl.figure()
    pl.violinplot(snps_per_sample, showmeans=True, vert=False, 
                  bw_method='silverman')
    pl.title('SNPs per sample')
    
    rand_ints = np.random.choice(seq_len, size=10, replace=False)
    print("Random SNP positions:\n\t{}".format(rand_ints))
    
    p_pool = rand_ints[:5]
    a_pool = rand_ints[5:]
    
    print("\tPositive SNPs: {}".format(p_pool))
    print("\tAnti-positive SNPs: {}".format(a_pool))
    
    variance_pool = [0, 10]
    print("Position variances:\n\t{}".format(variance_pool))
    
    def andc(nr_elements, start=0):
        return [[i] for i in range(start, start+nr_elements)]
    
    def orc(nr_elements, start=0):
        return [[i for i in range(start, start+nr_elements)]]
    
    samplesets = OrderedDict(\
                      A = dict(p = andc(1), a = []),
                      B = dict(p = andc(2), a = []),
                      C = dict(p = andc(5), a = []),
                      D = dict(p = andc(1), a = andc(1)),
                      E = dict(p = andc(5), a = andc(2)),
                      F = dict(p = orc(2), a = []),
                      G = dict(p = orc(2)+andc(1,3), a = orc(2))
                      )
    
    print("Sample combinatorics:\n\t{}".format("\n\t".join(\
        ["{} : {}".format(s, samplesets[s]) 
        for s in np.sort(list(samplesets.keys()))])))
    
    
    
    def make_dataset(variance, sset, set_nr, verbose=True):
        
        if verbose:
            print("Creating set {0} v{1}".format(sset, variance))
            sys.stdout.flush()
        
        # Set pool of coding SNPs
        all_p_snps = p_pool[np.fromiter(\
            chain.from_iterable(samplesets[sset]['p']), dtype=np.int)]
        all_a_snps = a_pool[np.fromiter(\
            chain.from_iterable(samplesets[sset]['a']), dtype=np.int)]
        
        # Set pool of non-coding SNPs
        non_coding_snps = np.arange(0,seq_len)
        non_coding_mask = np.ones_like(non_coding_snps, dtype=np.bool)
        for coding_snp in all_p_snps:
            non_coding_mask[coding_snp-variance:coding_snp+variance+1] = 0
        for coding_snp in all_a_snps:
            non_coding_mask[coding_snp-variance:coding_snp+variance+1] = 0
        non_coding_snps = non_coding_snps[non_coding_mask]
        
        # Create empty dataset (snps=SNPs in sample, unused=-1)
        data = dict(name=np.zeros(n_pos_seqs+n_neg_seqs, dtype=np.object), 
                    label=np.zeros(n_pos_seqs+n_neg_seqs, dtype='<i4'), 
                    snps=np.zeros((n_pos_seqs+n_neg_seqs, max_len), 
                                  dtype=np.int64)-1,
                    length=np.zeros(n_pos_seqs+n_neg_seqs, dtype=np.int64))
        
        # Initialize labels and names for pos and neg sequences
        data['label'][:n_pos_seqs] = 1
        data['name'][:n_pos_seqs] = 'p'
        data['name'][-n_pos_seqs:] = 'n'
            
        # Create samples
        for sample in np.arange(len(data['label'])):
            data['name'][sample] = data['name'][sample]+np.str(sample)+"_"
            
            if verbose:
                print("Creating sample {0}/{1}".format(sample, 
                      len(data['label'])), end='\r')
                sys.stdout.flush()
            
            # Create positive combination
            p_snps = np.fromiter(chain.from_iterable(\
                [np.random.choice(p, size=np.random.randint(low=1, 
                                                           high=len(p)+1),
                                replace=False)
                for p in samplesets[sset]['p']]), dtype=np.int)
            
            # Create anti-positive combination
            a_snps = np.fromiter(chain.from_iterable(\
                [np.random.choice(a, size=np.random.randint(low=1, 
                                                            high=len(a)+1), 
                                  replace=False) 
                for a in samplesets[sset]['a']]), dtype=np.int)
            
            # Pick coding snps to be added to sample
            if data['label'][sample] == 1: #positive sample
                # 50% chance for incomplete anti-positive snp set in positive sample
                if len(a_snps):
                    a_snps = np.random.choice(\
                                a_snps, 
                                size=np.random.randint(low=0, 
                                                       high=len(a_snps)), 
                                replace=False)
            else: # anti-positive sample
                if len(a_snps):
                    # if a anti-positive pattern exists 100/3% chance for no pattern, incomplete positive pattern, or positive+anti-positive pattern
                    antipick = np.random.choice([0,1,2]) 
                else:
                    # otherwise 50% chance to contain incomplete positive pattern
                    antipick = np.random.choice([0,1,2]) 
                    
                if antipick == 0:
                    # contain no pattern
                    a_snps = []
                elif antipick == 1:
                    # contain incomplete positive pattern and no anti-positive pattern
                    if len(p_snps):
                        p_snps = np.random.choice(\
                                    p_snps, 
                                    size=np.random.randint(low=0, 
                                                          high=len(p_snps)), 
                                    replace=False)
                    a_snps = []
                elif antipick == 3:
                    # contain positive pattern and anti-positive pattern
                    # add complete positive snp set to anti-positive sample
                    if len(p_snps):
                        p_snps = np.random.choice(\
                                    p_snps, 
                                    size=np.random.randint(low=0, 
                                                          high=len(p_snps)), 
                                    replace=False)
                    
            
            # Add indices to sample name
            data['name'][sample] = (data['name'][sample] + 
                                    "_".join(np.array(p_snps, dtype=str)) + 
                                    "_-_".join(np.array(a_snps, dtype=str)))
            # Coding snps to be added to sample
            cod_snps = np.append(p_pool[p_snps], a_pool[a_snps])
            # Nr of coding snps to be added to sample
            n_cod_snps = len(cod_snps)
            
            # Add uniform noise to coding snps
            if variance > 0:
                cod_snps += np.random.randint(low=-variance, 
                                              high=variance+1, 
                                              size=n_cod_snps)
            
            # Nr of non-coding SNPs in sample (distributed according to probs)
            n_ncod_snps = np.random.choice(a=x, p=probs) - n_cod_snps
            
            # Add non-coding snps to sample
            data['snps'][sample,:n_ncod_snps] =\
                np.random.choice(non_coding_snps, size=n_ncod_snps, 
                                 replace=False)
            
            # Add coding snps to sample
            data['snps'][sample,n_ncod_snps:n_ncod_snps+n_cod_snps] = cod_snps
            
            # Store nr of SNPs
            data['length'][sample] = n_ncod_snps+n_cod_snps
            
            
        
        # - data is ready at this point -
        
        
        # Save data to harddisk
        
        name = "{}_l{}_s{}_np{}_p{}_a{}_v{}".format(sset, seq_len, max_len,
                    n_pos_seqs,"_".join(str(x) for x in all_p_snps), 
                    " _".join(str(x) for x in all_a_snps), variance)
        name = os.path.join(dirname, name)
        
        if verbose:
            print("Saving {0}...".format(name+'.vcf.gz'))
            sys.stdout.flush()
        
        vcf_positions = np.unique(data['snps'])
        if vcf_positions[0] == -1:
            vcf_positions = vcf_positions[1:]
        
        if not large_vcf_file:
            
            temp_vcf = np.zeros((len(vcf_positions),
                                 9+data['snps'].shape[0]), dtype=np.object)
            temp_vcf[:,0] = 'chr'
            sys.stdout.flush()
            temp_vcf[:,3:9] = ['C','T','60','.','.','GT']
            sys.stdout.flush()
            temp_vcf[:,1] = vcf_positions
            temp_vcf[:,2] = np.arange(len(vcf_positions), dtpye=np.int64)
            
            temp_gt = np.zeros((seq_len), dtype=np.object)
            temp_mask = np.zeros_like(data['snps'][0], np.bool)
            for samp in np.arange(len(data['label'])):
                temp_mask = data['snps'][samp]!=-1
                temp_gt[:] = '0/0'
                temp_gt[data['snps'][samp][temp_mask]] = '1/1'
                temp_vcf[:,9+samp] = temp_gt[vcf_positions]
            
            with gzip.open(name+'.vcf.gz', 'wt', compresslevel=5) as vcf:
                vcf.write("\t".join(np.append(['#CHROM', 'POS', 'ID', 'REF', 
                                               'ALT', 'QUAL', 'FILTER', 
                                               'INFO', 'FORMAT'], 
                                               data['name'])))
                vcf.write("\n")
            
            with gzip.open(name+'.vcf.gz', 'ab', compresslevel=5) as vcf:
                np.savetxt(vcf,temp_vcf, delimiter='\t', newline='\n', 
                           fmt='%s')
                
        else: #large vcf file -> write line by line in junks
            junksize = 1e5
            num_junks = np.int32(np.ceil(len(vcf_positions) / junksize))
            junkrem = np.int32(np.mod(len(vcf_positions), junksize))
            
            # write header
            with gzip.open(name+'.vcf.gz', 'wt', compresslevel=5) as vcf:
                vcf.write("\t".join(np.append(['#CHROM', 'POS', 'ID', 'REF', 
                                               'ALT', 'QUAL', 'FILTER', 
                                               'INFO', 'FORMAT'], 
                                               data['name'])))
                vcf.write("\n")
            
            
            temp_vcf = np.zeros((junksize,9+data['snps'].shape[0]), 
                                dtype=np.object)
            temp_vcf[:,0] = 'chr'
            sys.stdout.flush()
            temp_vcf[:,3:9] = ['C','T','60','.','.','GT']
            sys.stdout.flush()
            
            # prepare by sorting snps
            data['snps'].sort(axis=1)
            temp_search_inds = np.zeros(junksize, dtype=np.int64)
            temp_search_mask = np.zeros(junksize, dtype=np.bool)
            
            for j in range(num_junks):
                if j < num_junks-1:
                    junkslice = slice(j*junksize, (j+1)*junksize)
                    junkend = junksize
                else:
                    junkslice = slice(j*junksize, j*junksize+junkrem)
                    junkend = junkrem
                
                # POS field
                temp_vcf[:junkend,1] = vcf_positions[junkslice]
                # ID
                temp_vcf[:,2] = np.arange(j*junksize, (j+1)*junksize)
                # initialize GTs
                temp_vcf[:junkend,9:] = '0/0'
                
                # loop through samples and set GTs at POS
                for samp in range(len(data['label'])):
                    # find positions closest to POS in sample
                    temp_search_inds[:junkend] =\
                        np.searchsorted(data['snps'][samp], 
                                        temp_vcf[:junkend,1])
                    # check where positions and POS overlap
                    temp_search_mask[:junkend] =\
                        np.take(data['snps'][samp], 
                                temp_search_inds[:junkend],
                                mode='clip') == temp_vcf[:junkend,1]
                    temp_search_mask[junkend:] = 0
                    # set GT at overlapping positions
                    temp_vcf[temp_search_mask, 9+samp] = '1/1'
                    
                with gzip.open(name+'.vcf.gz','ab',compresslevel=5) as vcf:
                    np.savetxt(vcf,temp_vcf[:junkend], delimiter='\t', 
                               newline='\n', fmt='%s')
                
        
        with open(name+'.info', 'wt') as csv:
            if verbose:
                print("Saving info file {0}...".format(name+'.info'))
                sys.stdout.flush()
            for sample in np.arange(len(data['label'])):
                csv.write(data['name'][sample] + ";" + 
                    str(data['label'][sample]) + ";" + 
                    str(data['length'][sample]) + ";" + 
                    ','.join(np.array(cod_snps, dtype=str)) + "\n")
                
        with open(name+'.csv', 'wt') as csv:
            if verbose:
                print("Saving label file {0}...".format(name+'.csv'))
                sys.stdout.flush()
            csv.write("" + "\t" + "p" + "\n")
            for sample in np.arange(len(data['label'])):
                csv.write(data['name'][sample] + "\t" + 
                    str(data['label'][sample]) + "\n")
    
    
    set_counter = 0
    
    print("\n")
    
    if multicore:
        from multiprocessing import Pool
        pool = Pool(processes=multicore)
    
    for variance in variance_pool:
        for sset in list(samplesets.keys()):
            if multicore:
                pool.apply_async(make_dataset,[variance, sset, 
                                               set_counter, verbose])
            else:
                make_dataset(variance, sset, set_counter, verbose)
            set_counter += 1
    
    if multicore:
        pool.close()
        pool.join()
    
    print("\nCreated dataset folder {}".format(dirname))

print("\nDone!")