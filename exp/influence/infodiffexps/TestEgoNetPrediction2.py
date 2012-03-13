
import numpy
import scipy.linalg
from apgl.util import *
from apgl.kernel import * 
from apgl.features.PrimalDualCCA import PrimalDualCCA
from apgl.predictors import *
from apgl.data import * 


"""
Lets try to figure out why we are getting such a bad prediction of the weight
matric for the information diffusion experiments. 
"""

outputDir = PathDefaults.getTempDir()
file = open(outputDir + "matrixFile.npz", 'rb')
arr = numpy.load(file)

Xe = arr["arr_0"]
W = arr["arr_1"]

print((Xe.shape))
print((numpy.min(Xe)))
print((numpy.max(Xe)))
print((numpy.linalg.norm(Xe)))
print((Util.rank(numpy.dot(Xe.T, Xe))))
print((numpy.linalg.cond(Xe)))

print((W.shape))
print((numpy.min(W)))
print((numpy.max(W)))
print((numpy.linalg.norm(W)))
print((Util.rank(numpy.dot(W.T, W))))
print((numpy.linalg.cond(W)))


D, V = scipy.linalg.eig(numpy.dot(Xe.T, Xe))
D2, V = scipy.linalg.eig(numpy.dot(W.T, W))

print((D.shape))
print((D2.shape))
print((numpy.min(D)))
print((numpy.max(D)))
print((numpy.min(D2)))
print((numpy.max(D2)))




#kernel = LinearKernel()
kernel = GaussianKernel()
kernel.setSigma(0.1)

standardiser = Standardiser()
standardiser2 = Standardiser()
Xe = standardiser.standardiseArray(Xe)
W = standardiser2.standardiseArray(W)

lmbda = 0.0001
predictor = KernelShiftRegression(kernel, lmbda)
A = predictor.learnModel(Xe, W)
W2 = predictor.predict(Xe)

print(Xe)
print(W)

print((numpy.linalg.norm(W)))
print((numpy.linalg.norm(W-W2)))