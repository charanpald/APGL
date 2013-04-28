
import numpy
import logging
import sys
import time 
import scipy.sparse
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.SVDUpdate import SVDUpdate
from exp.util.SparseUtils import SparseUtils 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SVDUpdateProfile(object):
    def __init__(self):
        #Create our matrices 
        numpy.random.seed(21)        
        
    def benchmark(self): 
        numMatrices = 20
        matrixList = [] 
        
        print("Generating matrices")        
        
        for i in range(numMatrices): 
            print("Iteration: " + str(i))
            m = numpy.random.randint(5000, 20000) 
            n = numpy.random.randint(5000, 20000) 
            density = numpy.random.rand()*0.1
            X = scipy.sparse.rand(m, n, density)
            
            r = numpy.random.randint(10, 50)
            U, s, V = SparseUtils.generateLowRank((m, n), r)
            
            print(m, n, density, r)
            matrixList.append((X, U, s, V))
        
        k = 10         
        
        times = [] 
        print("Starting timings for ARPACK")
        start = time.time() 
        
        for i, matrices in enumerate(matrixList): 
            print("Iteration: " + str(i))            
            X, U, s, V = matrices 
            SVDUpdate.addSparseArpack(U, s, V, X, k)
        
        times.append(time.time()-start)
        
        
        #Compare versus PROPACK 
        print("Starting timings for PROPACK")
        start = time.time() 
        
        for i, matrices in enumerate(matrixList): 
            print("Iteration: " + str(i))            
            X, U, s, V = matrices 
            SparseUtils.svdSparseLowRank(X, U, s, V, k)
        
        times.append(time.time()-start)
        print(times)
        
        #PROPACK is faster 

profiler = SVDUpdateProfile()
profiler.benchmark() 
