# -*- coding: utf-8 -*-
"""plotting_daemons.py: Functions for starting subprocesses for plotting


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

plotting_demon(): Spawns daemon subprocesses for plotting tasks in queue
start_plotting_daemon(): Creates a master subprocess who handles
    plotting_demon() to better deal with bugs and memory leaks of 
    theano GPU and matplotlib. Call this function before importing theano 
    and then send tuples containing function and lists of argument for 
    plotting through queue.


=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-30  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""
from .daemons import launch_proc
from multiprocessing import Pool

def plotting_demon(plotting_queue, multicore=3):
    """
    Queue plotting tasks and execute them parallel in daemon subprocesses
    
    Parameters
    -------
    plotting_queue : multiprocessing.Queue() instance
        Queue for plotting tasks; Elements in queue can be tuples with
        (function, list of arguments) or 0 for exit/termination of daemons
    multicore : int
        How many parallel daemons shall be spawned to work off the task-
        queue?
    """
    print("Starting printing daemon ...")
    pool = Pool(processes=multicore)
    while True:
        rec = plotting_queue.get()
        if rec == 0:
            pool.close()
            pool.join()
            del pool
            print("Printing daemon terminated.")
            exit(0)
        func, arguments = rec
        #launch_proc(func, arguments)
        pool.apply_async(func, arguments)
    
    pool.close()
    pool.join()
    del pool

def start_plotting_daemon(wait=False, multicore=3):
    """
    To better deal with bugs and memory leaks of theano GPU and matplotlib,
    start a full-fledged subprocess which handles the creation of plotting-
    daemons before theano is imported.
    
    Parameters
    -------
    wait : bool
        False: Do plotting in background and do not wait for completion of 
        piped function in plotting_queue before continuing;
        True: Wait for plotting function to finish before continuing
    multicore : int
        How many parallel daemons shall be spawned to work off the task-
        queue?
    
    Returns
    -------
    tuple (plotting_queue, proc)
    
    plotting_queue : multiprocessing.Queue() instance
        Queue for plotting tasks; Elements in queue can be tuples with
        (function, list of arguments) or 0 for exit/termination of daemons
    proc : mp.Process instance
        mp.Process instance that serves as handle for the subprocess
    Parameters
    -------
    Call this function before importing theano and then send tuples 
    containing function and lists of argument for plotting through queue
    """
    import multiprocessing as mp
    plotting_queue = mp.Queue()
    # Launch the master-subprocess which will handle the spawining of
    #  daemon subprocesses for plotting
    proc = launch_proc(target=plotting_demon, 
                       arguments=[plotting_queue, multicore], daemon=False,
                       wait=wait)
    return (plotting_queue, proc)