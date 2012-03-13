"""
A script to evaluate the error in centering using an artificially generated dataset.
Error is computed using
1/ell ||X - ju'|_f^2 = 1/ell tr(K) - 2/ell j'K alpha + alpha'K alpha
"""
import numpy.random
import numpy
from apgl.data.Preprocessor import Preprocessor
from apgl.kernel.LinearKernel import LinearKernel
from apgl.data.SparseCenterer import SparseCenterer
from apgl.util.Util import Util 

numpy.random.seed(21)

numExamples = 500
numFeatures = 100

j = numpy.ones((numExamples, 1))

centerVector = numpy.random.randn(numFeatures)

#Each example has about zero mean here 
X = numpy.random.random((numExamples, numFeatures)) - numpy.ones(numFeatures)*0.5
R = numpy.sqrt(numpy.max(sum(X**2, 0)))
X = X/R
print(("Norm squared of each centered example = " + str(numpy.max(sum(X**2, 0)))))
print(("Norm of the shift vector: " + str(numpy.linalg.norm(centerVector))))
X = X + centerVector
print(("Worst case of centering " + str(numpy.sqrt(numpy.sum(X**2)/numExamples))))

K = numpy.dot(X, X.T)

#Now, let's center the data using standard centering 
preprocessor = Standardiser()
Xc1 = preprocessor.centreArray(X)
Kc1 = numpy.dot(Xc1, Xc1.T)
alpha1 = j/numExamples

linearKernel = LinearKernel()
sparseCenterer = SparseCenterer()

error1 = numpy.trace(K)/numExamples - 2*Util.mdot(j.T, K, alpha1)/numExamples + Util.mdot(alpha1.T, K, alpha1)
#error1 = numpy.sqrt(error1)

step = int(numpy.floor(numExamples/20))
cs = numpy.array(list(range(step, numExamples, step)))
error2s = numpy.ones(cs.shape[0])

for i in range(cs.shape[0]):
    c = cs[i]
    Kc2, alpha2 = sparseCenterer.centerArray(X, linearKernel, c, True)
    #print(alpha2)
    error2s[i] = numpy.trace(K)/numExamples - 2*Util.mdot(j.T, K, alpha2)/numExamples + Util.mdot(alpha2.T, K, alpha2)
    #error2s[i] = numpy.sqrt(error2s[i])
    
print(error1)
print(cs)
print(error2s)

#Let's compute the error directly
error3 = numpy.linalg.norm(X - Util.mdot(j, alpha1.T, X), 'fro')**2/numExamples
print(error3)