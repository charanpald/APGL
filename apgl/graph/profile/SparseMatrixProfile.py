import numpy
import logging
import sys
import scipy.sparse
from cvxopt import spmatrix
from pysparse import spmatrix

from apgl.util import *


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class SparseMatrixProfile(object):
    def __init__(self):
        self.n = 10000
        self.m = 100000

    def profileCvxoptSparseAssign(self):
        W = spmatrix(0, [], [], size=(self.n,self.n))
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        def runAdd():
            for i in range(self.m):
                W[int(V[i,0]), int(V[i,1])] = u[i]

        ProfileUtils.profile('runAdd()', globals(), locals())

    def profileScipySparseAssign(self):
        W = scipy.sparse.lil_matrix((self.n, self.n))
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        def runAdd():
            for i in range(self.m):
                W[V[i,0], V[i,1]] = u[i]
            
        ProfileUtils.profile('runAdd()', globals(), locals())

    def profilePySparseAssign(self):
        W = spmatrix.ll_mat(self.n,self.n)
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        def runAdd():
            for i in range(self.m):
                W[int(V[i,0]), int(V[i,1])] = u[i]

        ProfileUtils.profile('runAdd()', globals(), locals())

    def profilePySparseKeys(self):
        """
        Test for memory leak 
        """
        W = spmatrix.ll_mat(self.n,self.n)
        V = numpy.random.randint(0, self.n, (self.m,2))
        u = numpy.random.rand(self.m)

        for i in range(self.m):
            W[int(V[i,0]), int(V[i,1])] = u[i]

        def runKeys():
            for i in range(self.m):
                for j in range(self.n):
                    neighbours = W[j, :]
                    #neighbours = W[j, :].keys()

        print("Running keys")
        runKeys()

profiler = SparseMatrixProfile()
#profiler.profileScipySparseAssign()
#profiler.profileCvxoptSparseAssign()
#profiler.profilePySparseAssign()
profiler.profilePySparseKeys()
#Spmatrix is about 15 times faster than scipy.sparse which is faster than cvxopt.spmatrix

#Also test retrieving edges, getting neighbours, getting nnz and degree. Submatrix.