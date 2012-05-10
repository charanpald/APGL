
"""
An implementation of the iterative SVDU-IPCA algorithm 
"""

import scipy 
import numpy
from apgl.util.Util import Util
from exp.sandbox.SVDUpdate import svdUpdate

numpy.set_printoptions(suppress=True, precision=3, linewidth=300)
numpy.random.seed(21)

numExamples1 = 15
numExamples2 = 10
numFeatures = 5
k = 4

#Note that the columns are the examples 
X1 = numpy.random.rand(numFeatures, numExamples1)
X2 = numpy.random.rand(numFeatures, numExamples2)
X = numpy.c_[X1, X2]

Xhat1 = X1 - numpy.outer(numpy.mean(X, 1), numpy.ones(numExamples1))
Xhat2 = X2 - numpy.outer(numpy.mean(X, 1), numpy.ones(numExamples2))
Xhat = numpy.c_[Xhat1, Xhat2]

sigma = numpy.dot(Xhat.T, Xhat)
sigma1 = numpy.dot(Xhat1.T, Xhat1)
sigma2 = numpy.dot(Xhat1.T, Xhat2)
sigma3 = numpy.dot(Xhat2.T, Xhat2)

d, U = numpy.linalg.eig(sigma1)
inds = numpy.flipud(numpy.argsort(d))
indsk = inds[0:k]

#rank k approximation of sigma 
sigma1k = numpy.dot(U[:, indsk], numpy.dot(numpy.diag(d[indsk]), U[:, indsk].T ))
ell = Util.rank(sigma1)

Ptilde1 = numpy.dot(numpy.diag(numpy.sqrt(d[indsk])), U[:, indsk].T)
Ptilde1 = numpy.r_[Ptilde1, numpy.zeros((ell-k, numExamples1))]

LambdaTildeSq = numpy.diag(d[inds[0:ell]] ** -0.5)
Utilde = U[:, inds[0:ell]]

Q1 = numpy.dot(LambdaTildeSq, numpy.dot(Utilde.T, sigma2))
Q2 = numpy.zeros((numExamples2, numExamples1))
#Q3 is zero which is odd 
Q3 = scipy.linalg.sqrtm(sigma3 - numpy.dot(Q1.T, Q1))

Ptilde2 = numpy.r_[Ptilde1, Q2]
Y = numpy.r_[Q1, Q3]

#Use the SVD update function 
A = Ptilde2
B = Y

U, s, V = numpy.linalg.svd(A)
V = V.T
inds = numpy.flipud(numpy.argsort(s))
U = U[:, inds[0:k]]
s = s[inds[0:k]]
V = V[:, inds[0:k]]

Utilde, Stilde, Vtilde = svdUpdate(U, s, V, B)
ABEst = numpy.dot(numpy.dot(Utilde, Stilde), Vtilde.T)
sigmaEst = numpy.dot(ABEst.T, ABEst)

print(numpy.real(sigmaEst))
print("\n\n\n\n")
print(sigma)

print(numpy.linalg.norm(sigma))
print(numpy.linalg.norm(sigmaEst - sigma))

#Seems to work! 
