import numpy
import logging
import sys
import scipy 
import scipy.sparse 
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.Nystrom import Nystrom 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class NystromProfile(object):
    def __init__(self):
        pass

    def profileEigpsd(self):
        n = 1000 
        p = 0.1 
        L = scipy.sparse.rand(n, n, p)            
        L = L.T.dot(L)
            
        cols = 500
        ProfileUtils.profile('Nystrom.eigpsd(L, cols)', globals(), locals())

profiler = NystromProfile()
profiler.profileEigpsd() 
