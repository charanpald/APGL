
"""
Some code to see if there is any pattern in the SVD of a matrix with fixed 
sparisty structure. 
"""
import sys 
import logging
import scipy.sparse
import numpy
from sparsesvd import sparsesvd
from exp.util.SparseUtils import SparseUtils 

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(precision=3, suppress=True, linewidth=100)

m = 10 
n = 10 
r = 1
U0, s0, V0 = SparseUtils.generateLowRank((m, n), r)

numInds = 10 
inds = numpy.unique(numpy.random.randint(0, m*n, numInds)) 
A = SparseUtils.reconstructLowRank(U0, s0, V0, inds)
#print(A.todense())

t0 = s0 + numpy.random.rand(s0.shape[0])*0.1
B = SparseUtils.reconstructLowRank(U0, t0, V0, inds)
#print(B.todense())

k = 9
U, s, V = sparsesvd(A, k)
U2, s2, V2 = sparsesvd(B, k)

print(A.todense())

print(U0)
print(s0)
print(V0)

print(U)
print(s)
print(V)


print(U2)
print(s2)
print(V2)

print(U2.T.dot(U))
#print(s2)
print(V2.T.dot(V))

#Now try for fixed singular vectors 