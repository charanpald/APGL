"""
Some code to test the SVD updating procedure
"""

import scipy
import numpy
from apgl.util.Util import Util

numpy.set_printoptions(suppress=True, precision=3, linewidth=300)
numpy.random.seed(21)

def svdUpdate(U, s, V, B):
    """
    Find the SVD of a matrix [A, B] where  A = U diag(s) V.T. 
    """
    if U.shape[0] != B.shape[0]:
        raise ValueError("U must have same number of rows as B")
    if s.shape[0] != U.shape[1]:
        raise ValueError("Number of cols of U must be the same size as s")
    if s.shape[0] != V.shape[1]:
        raise ValueError("Number of cols of V must be the same size as s")

    n = U.shape[0]
    k = U.shape[1]
    r = B.shape[1]
    m = V.shape[0]

    C = numpy.dot(numpy.eye(m) - numpy.dot(U, U.T), B)
    Q, R = numpy.linalg.qr(C)

    rPrime = Util.rank(C)
    Q = Q[:, 0:rPrime]
    R = R[0:rPrime, :]

    D = numpy.r_[numpy.diag(s), numpy.zeros((rPrime, k))]
    E = numpy.r_[numpy.dot(U.T, B), R]

    D = numpy.c_[D, E]

    Uhat, sHat, Vhat = numpy.linalg.svd(D)
    inds = numpy.flipud(numpy.argsort(sHat))
    Uhat = Uhat[:, inds[0:k]]
    sHat = sHat[inds[0:k]]
    Vhat = Vhat[inds[0:k], :].T

    #The best rank k approximation of [A, B]
    Utilde = numpy.dot(numpy.c_[U, Q], Uhat)
    Stilde = numpy.diag(sHat)

    G1 = numpy.r_[V, numpy.zeros((r, k))]
    G2 = numpy.r_[numpy.zeros((n ,r)), numpy.eye(r)]
    Vtilde = numpy.dot(numpy.c_[G1, G2], Vhat)

    return Utilde, Stilde, Vtilde
  
def svdUpdate2(U, s, V, B):
    """
    Find the SVD of a matrix [A, B] where  A = U diag(s) V.T. 
    """
    if U.shape[0] != B.shape[0]:
        raise ValueError("U must have same number of rows as B")
    if s.shape[0] != U.shape[1]:
        raise ValueError("Number of cols of U must be the same size as s")
    if s.shape[0] != V.shape[1]:
        raise ValueError("Number of cols of V must be the same size as s")

    n = U.shape[0]
    k = U.shape[1]
    r = B.shape[1]
    m = V.shape[0]

    C = numpy.dot(numpy.eye(m) - numpy.dot(U, U.T), B)
    Ubar, sBar, Vbar = numpy.linalg.svd(C)
    inds = numpy.flipud(numpy.argsort(sBar))
    Ubar = Ubar[:, inds[0:k]]
    sBar = sBar[inds[0:k]]
    Vbar = Vbar[inds[0:k], :].T

    rPrime = Ubar.shape[1]

    D = numpy.r_[numpy.diag(s), numpy.zeros((rPrime, k))]
    E = numpy.r_[numpy.dot(U.T, B), numpy.diag(sBar).dot(Vbar.T)]

    D = numpy.c_[D, E]

    Uhat, sHat, Vhat = numpy.linalg.svd(D)
    inds = numpy.flipud(numpy.argsort(sHat))
    Uhat = Uhat[:, inds[0:k]]
    sHat = sHat[inds[0:k]]
    Vhat = Vhat[inds[0:k], :].T

    #The best rank k approximation of [A, B]
    Utilde = numpy.dot(numpy.c_[U, Ubar], Uhat)
    Stilde = numpy.diag(sHat)

    G1 = numpy.r_[V, numpy.zeros((r, k))]
    G2 = numpy.r_[numpy.zeros((n ,r)), numpy.eye(r)]
    Vtilde = numpy.dot(numpy.c_[G1, G2], Vhat)

    return Utilde, Stilde, Vtilde

m = 10
n = 10
r = 5
k = 8
A = numpy.random.rand(m, n)
B = numpy.random.rand(m, r)

U, s, V = numpy.linalg.svd(A)
V = V.T
inds = numpy.flipud(numpy.argsort(s))
U = U[:, inds[0:k]]
s = s[inds[0:k]]
V = V[:, inds[0:k]]
Utilde, Stilde, Vtilde = svdUpdate2(U, s, V, B)
ABEst = numpy.dot(numpy.dot(Utilde, Stilde), Vtilde.T)

AB = numpy.c_[A, B] 

#Test that the approximation works 

U, s, V = numpy.linalg.svd(AB, full_matrices=False)
V = V.T 
inds = numpy.flipud(numpy.argsort(s))
U = U[:, inds[0:k]]
s = s[inds[0:k]]
V = V[:, inds[0:k]]

ABFull = numpy.dot(numpy.dot(U, numpy.diag(s)), V.T)

print(numpy.linalg.norm(AB))
print(numpy.linalg.norm(AB - ABFull))
print(numpy.linalg.norm(AB - ABEst))
print(numpy.linalg.norm(ABFull - ABEst))

#Works - hurray!