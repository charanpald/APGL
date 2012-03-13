"""
We want to see the spectrum of a 3 cluster example 
"""
import numpy

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200)

X1 = numpy.random.randn(5, 2) + numpy.array([0, 0])
X2 = numpy.random.randn(5, 2) + numpy.array([-15, 15])
X3 = numpy.random.randn(5, 2) + numpy.array([15, -15])

X = numpy.r_[X1, X2]
X = numpy.r_[X, X3]

X = X - numpy.mean(X, 0)
print(X)

K = numpy.dot(X, X.T)
W, V = numpy.linalg.eig(K)

print(W)
Khat = numpy.dot(K, V[:, [0,1]])
print(Khat)

P = numpy.dot(Khat, Khat.T)

print(P)
print((P>0))


