import scipy.sparse
import scipy.sparse.linalg 
import time
import numpy 
from apgl.util.Util import Util 

#Now test using scipy.sparse.linalg.lobpcg
#First compute the eigenvectors of a matrix, then modify it slightly and
#then use the same eigenvectors as approximations to the new ones

k = 50
n = 1000
X = scipy.sparse.rand(n, n, 0.01)
X = X.dot(X.T)
s, V = scipy.sparse.linalg.eigsh(X, k)

#Now change X a bit
dX = scipy.sparse.rand(n, n, 0.0001)
dX = dX.dot(dX.T)
print(dX.getnnz())
X = X + dX

startTime = time.time()
s1, V1 = scipy.sparse.linalg.eigs(X, k, which="LM")
timeTaken = time.time() - startTime
print(timeTaken)

#This function gives different results to the others
#In fact the eigenvalues are very different 
startTime = time.time()
s2, V2 = scipy.sparse.linalg.lobpcg(X, V, largest=True, maxiter=200, tol=10**-8, verbosityLevel=1)
timeTaken = time.time() - startTime
print(timeTaken)

#Now test with numpy
Xd = numpy.array(X.todense())
s3, V3 = numpy.linalg.eig(Xd)
inds = numpy.flipud(numpy.argsort(numpy.abs(s3)))
s3, V3 = Util.indEig(s3, V3, inds[0:k])

print(s1)
print(s2)
print(s3)

