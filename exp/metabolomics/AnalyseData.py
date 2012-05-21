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

YIgf1Inds, YICortisolInds, YTestoInds = MetabolomicsUtils.createIndicatorLabels(YList)

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

#Plot the correlation of the raw spectrum above x percent
Xr = numpy.random.rand(Xs.shape[0], Xs.shape[1])
datasets = [(Xr, "random"), (Xs, "raw"), (XwHaar, "haar"), (XwDb4, "db4"), (XwDb8, "db8")]

corLims = numpy.arange(0, 1.01, 0.01)

for dataset in datasets:
    X = Standardiser().standardiseArray(dataset[0])
    C = X.T.dot(X)

    w, V = numpy.linalg.eig(dataset[0].T.dot(dataset[0]))
    w = numpy.flipud(numpy.sort(w))

    correlations = numpy.zeros(corLims.shape[0])
    upperC = C[numpy.tril_indices(C.shape[0])]

    for i in range(corLims.shape[0]):
        correlations[i] = numpy.sum(numpy.abs(upperC) >= corLims[i])/float(upperC.size)


    plt.plot(corLims, correlations, label=dataset[1])


plt.xlabel("Absolute correlation lower bound")
plt.ylabel("Proportion of pairs")
plt.legend()
plt.show()

