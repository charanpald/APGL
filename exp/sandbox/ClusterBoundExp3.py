"""
We test the cluster bound on the spectral clustering approach 
""" 
import numpy 

#numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)

nrows = 10 
AA = numpy.random.randn(nrows, 5)
AA = AA.dot(AA.T)

U = numpy.random.randn(nrows, 2)
U = U.dot(U.T)

k = 3

lmbda, Q = numpy.linalg.eigh(AA)
sigma, P = numpy.linalg.eigh(U)

inds = numpy.flipud(numpy.argsort(lmbda)) 

indsk = inds[0:k]
Qk = Q[:, indsk] 
lmbdak = lmbda[indsk]

AAk = (Qk*lmbdak).dot(Qk.T)

lmbda2, Q2 = numpy.linalg.eigh(AA + U)
lmbda3, Q3 = numpy.linalg.eigh(AAk + U)

print(Q2.T.dot(Q3))

lmbda2 = numpy.sort(lmbda2)
lmbda3 = numpy.sort(lmbda3)


print(lmbda2)
print(lmbda3)