import numpy 
import scipy.sparse 
from apgl.graph import GraphUtils 
from apgl.util.Util import Util 

numpy.set_printoptions(suppress=True, precision=3)
n = 10
W1 = scipy.sparse.rand(n, n, 0.5).todense()
W1 = W1.T.dot(W1)
W2 = W1.copy()

W2[1, 2] = 1 
W2[2, 1] = 1  

print("W1="+str(W1))
print("W2="+str(W2))

L1 = GraphUtils.normalisedLaplacianSym(scipy.sparse.csr_matrix(W1))
L2 = GraphUtils.normalisedLaplacianSym(scipy.sparse.csr_matrix(W2))

deltaL = L2 - L1 


print("L1="+str(L1.todense()))
print("L2="+str(L2.todense()))
print("deltaL="+str(deltaL.todense()))

print("rank(deltaL)=" + str(Util.rank(deltaL.todense()))) 