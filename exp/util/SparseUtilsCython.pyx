from cython.operator cimport dereference as deref, preincrement as inc 
from sppy import csarray 
import struct
import numpy 
cimport numpy
import scipy.sparse 
 
numpy.import_array()

class SparseUtilsCython(object): 
    """
    Some Cythonised functions for sparse matrices. 
    """
    
    @staticmethod 
    def partialReconstructVals(numpy.ndarray[numpy.long_t, ndim=1] rowInds, numpy.ndarray[numpy.long_t, ndim=1] colInds, numpy.ndarray[numpy.float_t, ndim=2] U, numpy.ndarray[numpy.float_t, ndim=1] s, numpy.ndarray[numpy.float_t, ndim=2] V): 
        """
        Given an array of unique indices omega, partially reconstruct a matrix 
        using its SVD. 
        """ 
        cdef unsigned int i
        cdef unsigned int j 
        cdef unsigned int k
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] values = numpy.zeros(rowInds.shape[0], numpy.float)
        
        for i in range(rowInds.shape[0]):
            j = rowInds[i]
            k = colInds[i]
            
            values[i] = (U[j, :]*s).dot(V[k,:])            
            
        return values
        
    @staticmethod 
    def partialReconstruct(omega, U, s, V): 
        """
        Given an array of unique indices omega, partially reconstruct a matrix 
        using its SVD. The returned matrix is a scipy csc_matrix. 
        """ 
        X = csarray((U.shape[0], V.shape[0]), storageType="colMajor")
        X.reserve(omega[0].shape[0])
        for i in range(omega[0].shape[0]):
            j = omega[0][i]
            k = omega[1][i]
            
            X[j, k] = (U[j, :]*s).dot(V[k,:])            
            
        X.compress()
        return X.toScipyCsc()
        
    @staticmethod 
    def partialReconstruct2(omega, U, s, V): 
        """
        Given an array of unique indices omega, partially reconstruct a matrix 
        using its SVD. The returned matrix is a scipy csc_matrix. Uses Cython 
        to speed up the reconstruction. 
        """ 
        
        vals = SparseUtilsCython.partialReconstructVals(omega[0], omega[1], U, s, V)
        inds = numpy.c_[omega[0], omega[1]].T
        X = scipy.sparse.csc_matrix((vals, inds), shape=(U.shape[0], V.shape[0]))
        
        return X 
        