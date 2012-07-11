"""
Observe if C varies when we use more examples 
"""
import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from apgl.predictors.AbstractPredictor import computeTestError 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from apgl.predictors.LibSVM import LibSVM
import matplotlib.pyplot as plt 


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(45)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)
datasetName = datasets[9][0]

#sampleSizes = numpy.array([50, 100, 200])
sampleSizes = numpy.array([50, 100, 200])
foldsSet = numpy.arange(2, 13, 1)
alpha = 1.0

paramDict = {} 
paramDict["setC"] = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
paramDict["setGamma"] = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
paramDict["setEpsilon"] = numpy.array([2**-2])

sampleMethod = Sampling.crossValidation
numProcesses = multiprocessing.cpu_count()


j = 0 
trainX, trainY, testX, testY = ModelSelectUtils.loadRegressDataset(dataDir, datasetName, j)
learner = LibSVM(kernel='gaussian', type="Epsilon_SVR", processes=numProcesses) 


for sampleSize in sampleSizes: 
    print("Sample size " +str(sampleSize))
    trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
    validX = trainX[trainInds,:]
    validY = trainY[trainInds]
    
    folds = 5 
    idx = sampleMethod(folds, sampleSize)
    
    meanErrors = learner.parallelPenaltyGrid(validX, validY, testX, testY, paramDict, computeTestError)
    meanErrors = numpy.squeeze(meanErrors)
    
    for i in range(paramDict["setGamma"].shape[0]): 
        plt.figure(i)
        plt.plot(numpy.arange(paramDict["setC"].shape[0]), meanErrors[i, :], label=str(sampleSize))
        plt.legend(loc="upper left")
        plt.xlabel("C")
        plt.ylabel("Error")
plt.show()