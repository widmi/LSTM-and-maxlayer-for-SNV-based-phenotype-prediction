# -*- coding: utf-8 -*-
"""external_sources.py: Functions from external sources


Author -- Michael Widrich
Created on -- 2016-07-28
Contact -- k1257264@jku.students.at

Functions from external sources (code-snippets from forums etc.)

=======  ==========  =================  ===================================
Version  Date        Author             Description
1.0      2016-07-30  Michael Widrich    Added more comments and prepared
                                        for github
=======  ==========  =================  ===================================
"""

import os
import errno
import numpy as np

#### taken from external sources:

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def make_sure_path_exists(path):
    '''
    Create a path, if it does not already exist
    '''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class Tee(object):
    '''
    Redirect sys.stdout to multiple files 
    '''
    def __init__(self, original_stdout, *files):
        self.files = files
        self.original_stdout = original_stdout
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        try:
            self.original_stdout.flush()
        except AttributeError:
            # if original_stdout does not support flush()
            pass

class ConfusionMatrix:
    """
       Simple confusion matrix class for calculation of differen performance
       measures; row is the true class, column is the predicted class;
    """
    
    def __init__(self, n_classes, flat=False):
        self.n_classes = n_classes
        self.mat = np.zeros((n_classes,n_classes),dtype='int')
        self.flat = flat

    def __str__(self):
        return np.array_str(self.mat)

    def batchAdd(self,y_true,y_pred):
        assert len(y_true) == len(y_pred)
        assert max(y_true) < self.n_classes
        assert max(y_pred) < self.n_classes
        for i in range(len(y_true)):
                self.mat[y_true[i],y_pred[i]] += 1

    def zero(self):
        self.mat.fill(0)

    def getErrors(self):
        """
        Calculate differetn error types
        """
        tp = np.asarray(np.diag(self.mat).flatten(),
                        dtype='float')
        fn = np.asarray(np.sum(self.mat, axis=1).flatten(),
                        dtype='float') - tp
        fp = np.asarray(np.sum(self.mat, axis=0).flatten(),
                        dtype='float') - tp
        tn = np.asarray(np.sum(self.mat) * 
            np.ones(self.n_classes).flatten(), dtype='float') - tp - fn - fp
        if self.flat:
            return tp[1], tn[1], fp[1], fn[1]
        return tp, tn, fp, fn

    def accuracy(self):
        tp, tn, fp, fn = self.getErrors()
        return (tp + tn) / (tp + tn + fp + fn)

    def balanced_accuracy(self):
        return (self.sensitivity() + self.specificity()) / 2
    
    def sensitivity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fn)
        res = res[~np.isnan(res)]
        return res

    def specificity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fp)
        res = res[~np.isnan(res)]
        return res

    def gmean(self):
        return np.sqrt(self.sensitivity() * self.specificity())
    
    def positivePredictiveValue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def negativePredictiveValue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fn)
        res = res[~np.isnan(res)]
        return res

    def falsePositiveRate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (fp + tn)
        res = res[~np.isnan(res)]
        return res

    def falseDiscoveryRate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def F1(self):
        tp, tn, fp, fn = self.getErrors()
        res = (2*tp) / (2*tp + fp + fn)
        res = res[~np.isnan(res)]
        return res

    def matthewsCorrelation(self):
        tp, tn, fp, fn = self.getErrors()
        numerator = tp*tn - fp*fn
        denominator = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        res = numerator / denominator
        res = res[~np.isnan(res)]
        return res
    def getMat(self):
        return self.mat
