import numpy
import logging
import sys
import scipy 
import scipy.sparse 
import scipy.sparse.linalg 
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.RandomisedSVD import RandomisedSVD 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class RandomisedSvdProfile(object):
    def __init__(self):
        numpy.random.seed(21)

    def profileSvd(self):
        n = 5000 
        p = 0.1 
        L = scipy.sparse.rand(n, n, p)            
        L = L.T.dot(L)
            
        k = 50 
        q = 2
        ProfileUtils.profile('RandomisedSVD.svd(L, k, q)', globals(), locals())
        
        #Compare against the exact svd 
        #ProfileUtils.profile('scipy.sparse.linalg.svds(L, k=2*k)', globals(), locals())

profiler = RandomisedSvdProfile()
profiler.profileSvd() #51.4 s
