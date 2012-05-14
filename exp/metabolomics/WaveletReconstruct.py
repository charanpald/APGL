#Find the wavelet reconstruction of the data 
import logging
import numpy
import sys
import pywt 
from exp.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from apgl.util.PathDefaults import PathDefaults
from rpy2.robjects.packages import importr
from socket import gethostname
import matplotlib.pyplot as plt
from apgl.data.Standardiser import Standardiser

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)
numpy.set_printoptions(linewidth=160, precision=3, suppress=True)

treeRankLib = importr('TreeRank')
baseLib = importr('base')
baseLib.options(warn=1)

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
X, X2, Xs, XOpls, YList, ages, df = MetabolomicsUtils.loadData()

waveletStr = 'db4'
mode = "cpd"
maxLevel = 10
errors = numpy.zeros(maxLevel)
numFeatures = numpy.zeros(maxLevel)

level = 10 
waveletStrs = ["haar", "db4", "db8"]

#The variances are very similar across different wavelets 
for waveletStr in waveletStrs:
    Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
    standardiser = Standardiser()
    Xw = standardiser.centreArray(Xw)
    w, V = numpy.linalg.eig(Xw.dot(Xw.T))
    w = numpy.flipud(numpy.sort(w))

    variances = []
    variances.append(numpy.sum(w[0:1])/numpy.sum(w))
    variances.append(numpy.sum(w[0:5])/numpy.sum(w))
    variances.append(numpy.sum(w[0:10])/numpy.sum(w))
    variances.append(numpy.sum(w[0:15])/numpy.sum(w))
    variances.append(numpy.sum(w[0:20])/numpy.sum(w))
    variances.append(numpy.sum(w[0:25])/numpy.sum(w))
    variances.append(numpy.sum(w[0:50])/numpy.sum(w))
    variances.append(numpy.sum(w[0:100])/numpy.sum(w))
    variances.append(numpy.sum(w[0:150])/numpy.sum(w))
    variances.append(numpy.sum(w[0:200])/numpy.sum(w))
    #print(variances)


#100 PCs models 0.9908% of variance
plt.figure(2)
plt.plot(range(100), w[0:100], "k")
plt.xlabel("Eigenvalue rank")
plt.ylabel("Eigenvalue")
#plt.show()


#Now try some filtering and plot N versus reconstruction error
Ns = range(0, 700, 50)
waveletStrs = ['haar', 'db4', 'db8']
errors = numpy.zeros((len(waveletStrs), len(Ns)))
mode = "cpd"

standardiser = Standardiser()
#X = standardiser.centreArray(X)

for i in range(len(waveletStrs)):
    waveletStr = waveletStrs[i]
    Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
    C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)

    for j in range(len(Ns)):
        N = Ns[j]
        Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)
        X2 = MetabolomicsUtils.reconstructSignal(X, Xw2, waveletStr, mode, C)

        errors[i, j] = numpy.linalg.norm(X - X2)

#Plot example wavelet after filtering 
waveletStr = "haar"
N = 100
Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)
X2 = MetabolomicsUtils.reconstructSignal(X, Xw2, waveletStr, mode, C)

plt.figure(3)
plt.plot(range(X.shape[1]), X[0, :])
plt.plot(range(X.shape[1]), X2[0, :])

plt.figure(4)
for i in range(errors.shape[0]):
    plt.plot(Ns, errors[i, :], label=waveletStrs[i])
    plt.xlabel("N")
    plt.ylabel("Error")

print(errors)

plt.legend()
plt.show()