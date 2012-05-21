"""
Let's test functional treerank versus ordinary TreeRank with filtering on the
Synthetic Control Chart Time Series data
"""

from apgl.util.PathDefaults import PathDefaults
from exp.metabolomics.MetabolomicsUtils import MetabolomicsUtils
import numpy
import pywt 


dataDir = PathDefaults.getDataDir() + "functional/"
fileName = dataDir + "synthetic_control.data"

X = numpy.loadtxt(fileName)

#Ignore first 200 examples
X = X[200:, :]
Y = numpy.zeros(X.shape[0])
Y[0:200] = -1 #Increading trend and decreasing trend
Y[200:] = 1 #Upward shift and downward shift

#Compute wavelets

waveletStr = "db2"
level = 2
mode = "cpd"
Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)

print(X.shape)
print(Xw.shape)

C = pywt.wavedec(X[0, :], waveletStr, mode, level)

for c in C:
    print(c.shape)