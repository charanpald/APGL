
#Test if we can easily get the SVD of a set of matrices with low rank but under 
#a fixed structure 

import numpy 
import scipy.sparse 
from exp.util.SparseUtils import SparseUtils 

numpy.set_printoptions(suppress=True, precision=3, linewidth=150)

shape = (15, 20)
r = 10
k = 50
X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)

X = numpy.array(X.todense())

Y = numpy.zeros(X.shape)
Y[X.nonzero()] = 1

print(Y)

U2, s2, V2 = numpy.linalg.svd(Y)
print(s2)

X2 = numpy.zeros(X.shape)
for i in range(r): 
    X2 += s[i]*numpy.diag(U[:,i]).dot(Y).dot(numpy.diag(V[:, i]))

