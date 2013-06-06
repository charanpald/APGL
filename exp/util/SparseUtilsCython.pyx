from cython.operator cimport dereference as deref, preincrement as inc 
import cython
import struct
import numpy 
cimport numpy
import scipy.sparse 
numpy.import_array()

cdef extern from "SparseUtilsCython.cpp": 
    void partialReconstructValsPQCpp(int*, int*, double*, double*, double*, int, int) 

class SparseUtilsCython(object): 
    """
    Some Cythonised functions for sparse matrices. 
    """
        
    @staticmethod 
    def partialReconstructValsPQ(numpy.ndarray[int, ndim=1] rowInds, numpy.ndarray[int, ndim=1] colInds, numpy.ndarray[double, ndim=2, mode="c"] P, numpy.ndarray[double, ndim=2, mode="c"] Q): 
        """
        Given an array of unique indices inds, partially reconstruct $P*Q^T$. Do 
        the heavy work in C++. 
        """ 
        if P.shape[1] != Q.shape[1]: 
            raise ValueError("Matrices not aligned")
        
        cdef numpy.ndarray[double, ndim=1, mode="c"] values = numpy.zeros(rowInds.shape[0])
        partialReconstructValsPQCpp(&rowInds[0], &colInds[0], &P[0,0], &Q[0,0], &values[0], rowInds.shape[0], P.shape[1])          
        return values        

    @staticmethod 
    def partialReconstructPQ(omega, P, Q): 
        """
        Given an array of unique indices inds, partially reconstruct $P*Q^T$.
        The returned matrix is a scipy csc_matrix.
        """ 
        rowInds = numpy.array(omega[0], numpy.int32)
        colInds = numpy.array(omega[1], numpy.int32)
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, P, Q)
        inds = numpy.c_[omega[0], omega[1]].T
        X = scipy.sparse.csc_matrix((vals, inds), shape=(P.shape[0], Q.shape[0]))
        
        return X      
     
    @staticmethod
    def partialOuterProduct(numpy.ndarray[numpy.long_t, ndim=1] rowInds, numpy.ndarray[numpy.long_t, ndim=1] colInds, numpy.ndarray[numpy.float_t, ndim=1] u, numpy.ndarray[numpy.float_t, ndim=1] v):
        """
        Given an array of unique indices omega, partially reconstruct a matrix 
        using two vectors u and v 
        """ 
        cdef unsigned int i
        cdef unsigned int j 
        cdef unsigned int k
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] values = numpy.zeros(rowInds.shape[0], numpy.float)
        
        for i in range(rowInds.shape[0]):
            j = rowInds[i]
            k = colInds[i]
            
            values[i] = u[j]*v[k]            
            
        return values    
    

        