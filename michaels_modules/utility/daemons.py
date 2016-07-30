# -*- coding: utf-8 -*-
"""daemons.py: Functions for starting multiprocess deamons


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-30  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

def launch_proc(target, arguments, wait=False, daemon=True):
    """
    Execute function target in new subprocess
    
    Parameters
    -------
    target : function
        Function to launch
    arguments : list
        Arguments for target function
    wait : bool
        Wait for function to finish?
    daemon : bool
        Start subprocess as a deamon?
    Returns
    -------
    proc : mp.Process instance
        mp.Process instance that serves as handle for the subprocess
    """
    import multiprocessing as mp
    proc = mp.Process(target=target, args=arguments)
    proc.daemon = daemon
    proc.start()
    if wait:
        proc.wait()
    return proc