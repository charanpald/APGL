
import numpy
import logging
import sys
import scipy.sparse.linalg
import scipy.io
from apgl.util.ProfileUtils import ProfileUtils
from exp.util.SparseUtils import SparseUtils
from apgl.util.PathDefaults import PathDefaults 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SparseUtilsProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
    def profilePropackSvd(self):
        dataDir = PathDefaults.getDataDir() + "erasm/contacts/" 
        trainFilename = dataDir + "contacts_train"        
        
        trainX = scipy.io.mmread(trainFilename)
        trainX = scipy.sparse.csc_matrix(trainX, dtype=numpy.int8)
        
        
        k = 500 
        U, s, V = SparseUtils.svdPropack(trainX, k, kmax=k*5)
        
        print(s)
        
        #Memory consumption is dependent on kmax
        print("All done")
        
profiler = SparseUtilsProfile()
profiler.profilePropackSvd()
