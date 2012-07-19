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
from apgl.util import Evaluator 
import sklearn.metrics 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(45)
numpy.set_printoptions(linewidth=150)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)
datasetName = datasets[9][0]

#sampleSizes = numpy.array([50, 100, 200])
sampleSizes = numpy.array([50, 100, 150, 200, 250, 300])
foldsSet = numpy.arange(2, 13, 1)
alpha = 1.0

Cs = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
gammas = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
epsilons = numpy.array([2**-2])

paramDict = {} 
paramDict["setC"] = Cs 
paramDict["setGamma"] = gammas
paramDict["setEpsilon"] = epsilons

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
        
    #errors = learner.parallelPenaltyGrid(validX, validY, testX, testY, paramDict, computeTestError)
    #errors = numpy.squeeze(errors)
    
    errors = numpy.zeros((Cs.shape[0], gammas.shape[0]))
    norms = numpy.zeros((Cs.shape[0], gammas.shape[0]))
    
    for i, C in enumerate(Cs): 
        for j, gamma in enumerate(gammas):
            learner.setEpsilon(epsilons[0])
            learner.setC(C)
            learner.setGamma(gamma)
            learner.learnModel(validX, validY)
            predY = learner.predict(testX)
            errors[i, j] = Evaluator.meanAbsError(predY, testY)
            norms[i, j] = learner.weightNorm()
            
    
    for i in range(gammas.shape[0]): 
        plt.figure(i)
        plt.plot(numpy.log(Cs), errors[:, i], label=str(sampleSize))
        plt.legend(loc="upper left")
        plt.xlabel("C")
        plt.ylabel("Error")
        
        plt.figure(i+gammas.shape[0])
        plt.plot(norms[:, i], errors[:, i], label=str(sampleSize))
        plt.legend(loc="upper left")
        plt.xlabel("Norm")
        plt.ylabel("Error")
        
plt.show()