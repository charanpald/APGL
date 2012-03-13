import sys
import os
import numpy
import logging
import multiprocessing
from apgl.predictors.LibSVM import LibSVM
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.util.FileLock import FileLock
from apgl.util.Sampling import Sampling
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.modelselect.ModelSelectUtils import ModelSelectUtils, computeIdealPenalty, parallelPenaltyGridRbf
import matplotlib.pyplot as plt

"""
We want to compare the ideal and approximated penalty on the toy data 
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

sampleSize = 50
folds = 10
cvScalings = numpy.array([1.0])
sampleMethods = [("CV", Sampling.crossValidation), ("BS", Sampling.bootstrap), ("SS", Sampling.shuffleSplit)]
numProcesses = multiprocessing.cpu_count()

logging.debug("Running " + str(numProcesses) + " processes")
logging.debug("Process id: " + str(os.getpid()))

datasetNames = []
datasetNames.append(("toyData", 20))

dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"

i = 0 
datasetName = datasetNames[i][0]
numRealisations = datasetNames[i][1]
logging.debug("Learning using dataset " + datasetName)

data = numpy.load(dataDir + datasetName + ".npz")
gridPoints, trainX, trainY, pdfX, pdfY1X, pdfYminus1X = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]

#We form a test set from the grid points
testX = numpy.zeros((gridPoints.shape[0]**2, 2))
for m in range(gridPoints.shape[0]):
    testX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 0] = gridPoints
    testX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 1] = gridPoints[m]

svm = LibSVM()

logging.debug("Using sample size " + str(sampleSize) + " and " + str(folds) + " folds")

perm = numpy.random.permutation(trainX.shape[0])
trainInds = perm[0:sampleSize]
validX = trainX[trainInds, :]
validY = trainY[trainInds]
logging.debug("Finding ideal grid of penalties")
idealGrid = parallelPenaltyGridRbf(svm, validX, validY, testX, gridPoints, pdfX, pdfY1X, pdfYminus1X)

for s in range(len(sampleMethods)):
    sampleMethod = sampleMethods[s][1]
    logging.debug("Sampling method :" + str(sampleMethod))
    
    idx = sampleMethod(folds, validY.shape[0])
    svmGridResults = svm.parallelVfPenRbf(validX, validY, idx, cvScalings)

    bestSVM, trainErrors, approxGrid = svmGridResults[0]
    meanErrors = trainErrors + approxGrid

    plt.figure(s)
    plt.title("Sampling method :" + str(sampleMethod))
    plt.scatter(idealGrid.flatten(), approxGrid.flatten())
    plt.xlabel("Ideal penalty")
    plt.ylabel("Approximate penalty")

    plt.figure(s + len(sampleMethods))
    plt.title("Sampling method :" + str(sampleMethod))
    plt.contourf(numpy.log2(svm.gammas), numpy.log2(svm.Cs), numpy.abs(idealGrid - approxGrid), 100, antialiased=True)
    plt.colorbar()
    plt.xlabel("Gamma")
    plt.ylabel("C")

    plt.figure(s + 2*len(sampleMethods))
    plt.title("Sampling method :" + str(sampleMethod))
    plt.contourf(numpy.log2(svm.gammas), numpy.log2(svm.Cs), meanErrors, 100, antialiased=True)
    plt.colorbar()
    plt.xlabel("Gamma")
    plt.ylabel("C")

plt.show()