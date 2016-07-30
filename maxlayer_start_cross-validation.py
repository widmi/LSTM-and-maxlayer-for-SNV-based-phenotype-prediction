# -*- coding: utf-8 -*-
"""maxlayer_start_cross-validation.py: Specify parameter settings for
    cross-validation


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Specify parameter settings for simple k-fold cross-validation, then run this
file. Number of parallel processes and device for computations can also be
set here.


=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-28  Michael Widrich    Created as simple example
=======  ==========  =================  ===================================
"""

import os
import multiprocessing 
import time
import itertools
from michaels_modules.utility.external_sources import make_sure_path_exists

def start_experiment_process(exp_args):
    import os
    # select device for optimization by theano ('cpu', 'gpu', 'gpu0', ...)
    os.environ["THEANO_FLAGS"] = "device=gpu"
    # start and perform the experiment
    from main_maxlayer import start_experiment
    start_experiment(**exp_args)



if __name__ == '__main__':
    
    # number of parallel processes (0 for no multi-processing)
    num_parallel = 0
    # semaphore pool for subprocess management
    semaphore_pool = dict(num_parallel=\
                            multiprocessing.Semaphore(num_parallel or 1))
    # number of folds of cross validation
    folds = 5
    # mini-batch size
    batch_size = 20
    
    # number of training epochs to try
    eps = [2000]
    
    # netconfigurations to try
    netconfig = [
                 dict(tc_layer=20, # number of TC layer nodes
                      tc='max',  # TC layer as maxlayer 'max' or meanlayer'mean'
                      dense=[int(20/3)], # size of dense layer(s) [4,5] -> 2 dense layers with 4 and 5 nodes
                      act='elu', # activation functions in TC layer
                      dact='identity', # activation functions in dense layer
                      input_encoder='binary', # input encoder (see LSTM skripts for triangle example)
                      input_do=0.25 # input dropout rate
                      ),
                 dict(lstm=[], tc_layer=20, tc='max', dense=[int(20/3)], act='elu',  dact='identity', input_encoder='binary'),
                ]
    
    # updatespecifications to try
    updatespecs = [\
                   dict(alg = 'adadelta',  # update algorithm
                        regularization = dict(function='l2', # L2 weight penalty
                                              weights=[1e-3,1e-3,1e-3], # weights for L2 penalty [TC layer, dense layer, output layer]
                                              l2_fanout=0,
                                              nwp=1e-3 # L2 penalty weight for negative weights to input layer
                                              ),
                        loss_function = 'binary_crossentropy',
                        constraint=False, # constrain sum of absolute weight values to maximum value
                        force_sum=False # norm weights to constant sum of absolute values
                        ),
                   dict(alg = 'adadelta', 
                        regularization = dict(function='l2',
                                              weights=[1e-2,1e-2,1e-2],
                                              l2_fanout=0,
                                              ts_pos_weights=1e-3),
                        loss_function = 'binary_crossentropy',
                        constraint=False, force_sum=False),
                        ]
    
    
    try:
        # locations of input files and target files
        datapath = "data/"
        datafiles = [dict(input_file=os.path.join(datapath, "20160215_SNP_Artificial_3/B_l6500000_s5850_np100_p2730873_5490525_a_v0.hdf5"), 
                          target_file=os.path.join(datapath, "20160215_SNP_Artificial_3/B_l6500000_s5850_np100_p2730873_5490525_a_v0.csv"),),
                     dict(input_file=os.path.join(datapath, "20160215_SNP_Artificial_3/B_l6500000_s5850_np100_p2730873_5490525_a_v10.hdf5"), 
                          target_file=os.path.join(datapath, "20160215_SNP_Artificial_3/B_l6500000_s5850_np100_p2730873_5490525_a_v10.csv"),)]
        
        # location of folder to write results into 
        results_path = "Experiments_1/"
        make_sure_path_exists(results_path)
        
        # create a list with all combinations of the parameters, datasets,
        #  and inner cross-validation folds to test
        experiments = list(itertools.product(range(0, folds), eps, 
                                             list(range(len(netconfig))),
                                             list(range(len(updatespecs))),
                                             datafiles))
        
        # create an index file for the experiments and their settings
        with open(os.path.join(results_path,"experiments_index.csv"), "a") as indexfile:
            indexfile.write("folder name;experiment;net configuration;update algorithm;minibatch size\n")
            for p in experiments:
                if netconfig[p[2]].get('add_binary_encoding', False):
                    batch_size = batch_size
                else:
                    batch_size = batch_size
                indexfile.write('_'.join(str(x) for x in p[2:-1]) + ";")
                indexfile.write('{}/{}/{}'.format(p[0],os.path.basename(p[-1]['input_file']).split('.')[0], '_'.join(str(x) for x in p[2:-1])) + ";")
                indexfile.write("{}".format(netconfig[p[2]]) + ";")
                indexfile.write("{}".format(updatespecs[p[3]]) + ";")
                indexfile.write("{}".format(batch_size) + "\n")
        
        # loop through and start the experiments
        for p_i, p in enumerate(experiments):
            
            # number of experiment (ID)
            exp_nr = p_i
            # acquire semaphore or wait for one
            semaphore_pool['num_parallel'].acquire(block=True, timeout=None)
            
            print(">>Starting process for experiment {} / {}...".format(exp_nr, len(experiments)))
            
            # Create name for experiment (for saving and plotting networks)
            name = '{}/{}/{}'.format(p[0],os.path.basename(p[-1]['input_file'][:-len('.hdf5')]), '_'.join(str(x) for x in p[2:-1]))
            name = os.path.join(results_path, name)
            make_sure_path_exists(name)
            
            # Create dicionary with parameters for experiment
            exp_args = dict(exp_nr=exp_nr, # experiment ID
                            num_epochs=p[1], # number of training epochs
                            batch_size=batch_size, # size of minibatches
                            name=name, # name of experiment
                            datafile=p[4], # dict with datafile locations
                            updatespec=updatespecs[p[3]], # dict (see above)
                            layerspec=netconfig[p[2]], # dict (see above)
                            loadname=None, # location of trained network to load
                            fold=p[0], # current cross-validation fold
                            folds=folds, # number of cross-validation folds
                            semaphore_pool=semaphore_pool,
                            print_to_console=False # print to console?
                            )
            
            if num_parallel > 1:
                # spawn subprocess for experiment
                proc = multiprocessing.Process(target=start_experiment_process, args=(exp_args,))
                
                proc.daemon = False
                proc.start()
                    
                # join processes if they have finished
                active_children = multiprocessing.active_children()
                print(">>Running processes: {} ...".format(len(active_children)))
                time.sleep(30)
            else:
                # start experiment without subprocess (warning: might lead
                #  to aggregating memory leaks on GPU)
                exp_args['print_to_console'] = True
                start_experiment_process(exp_args)
        
        
        # Get last running processes
        active_children = multiprocessing.active_children()
        # And wait for them to finish
        for active_child in active_children:
            active_child.join()
    
        print(">>Done!")
    except:
        # Get last running processes
        active_children = multiprocessing.active_children()
        # And wait for them to finish
        for active_child in active_children:
            active_child.join()
        print(">>An error occured but child processes could be joined!")
        raise

