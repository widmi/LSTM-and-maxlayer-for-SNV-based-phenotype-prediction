# -*- coding: utf-8 -*-
"""convenience.py: Some multi-purpose utilities


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at


=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-30  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""
    
def remove_difficult_chars(string, chars=['(',')','-'], repl=''):
    """replace characters in list chars with repl in a string
    """
    string = string.replace(' ', '_')
    for char in chars:
        string = string.replace(char, repl)
    return string