from cython.parallel cimport prange

import numpy 
cimport numpy 

def foo(int m, int n, int p):
    cdef unsigned int i, numJobs, cols

    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] X = numpy.random.rand(m, n)
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] W = numpy.random.rand(m, n)
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] P = numpy.zeros((m, n)) 
    
    numJobs = 8 
    colInds = numpy.array(numpy.linspace(0, W.shape[1], numJobs+1), numpy.int) 
    
    for j in prange(n, nogil=True): 
        with gil: 
            for i in prange(m, nogil=True):
            
            
                #P[:, colInds[i]:colInds[i+1]] = X.dot(W[:, colInds[i]:colInds[i+1]])
                P[i, j] = X[i, j] + W[i, j]

    return P