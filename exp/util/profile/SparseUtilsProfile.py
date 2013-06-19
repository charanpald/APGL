
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

    def profileArpackSvd(self):
        dataDir = PathDefaults.getDataDir() + "erasm/contacts/" 
        trainFilename = dataDir + "contacts_train"        
        
        trainX = scipy.io.mmread(trainFilename)
        trainX = scipy.sparse.csc_matrix(trainX, dtype=numpy.float32)
        print(trainX.dtype.char, trainX.dtype)
        
        
        k = 500 
        U, s, V = SparseUtils.svdArpack(trainX, k, kmax=k*5)
        
        print(s)
        
        #Memory consumption is dependent on kmax and less than PROPACK 
        print("All done")

    def profileSubmatrix(self): 
        shape = (100000, 15000) 
        r = 50
        k = 5000000
    
        X = SparseUtils.generateSparseLowRank(shape, r, k) 
        print(X.nnz, type(X))
        
        inds = numpy.random.permutation(X.nnz)[0:1000000]
        
        ProfileUtils.profile('SparseUtils.submatrix(X, inds)', globals(), locals()) 

profiler = SparseUtilsProfile()
#profiler.profilePropackSvd()
#profiler.profileArpackSvd()
profiler.profileSubmatrix()
