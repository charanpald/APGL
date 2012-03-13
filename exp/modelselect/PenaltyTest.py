#Figure out whether the penalty increases with the number of folds 


import multiprocessing
import sys
from apgl.predictors.LibSVM import LibSVM, computeTestError
from apgl.predictors.DecisionTree import DecisionTree
from apgl.util.FileLock import FileLock
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator
from apgl.util.Util import Util
from apgl.modelselect.ModelSelectUtils import ModelSelectUtils
import logging
import numpy
import os


datasets = ModelSelectUtils.getRegressionDatasets()

numProcesses = 8
dataDir = PathDefaults.getDataDir() + "modelPenalisation/regression/"
datasetName = datasets[9]
print(datasetName)

j = 0 
trainX, trainY, testX, testY = ModelSelectUtils.loadRegressDataset(dataDir, datasetName, j)

learner = LibSVM(kernel='gaussian', type="Epsilon_SVR", processes=numProcesses) 


paramDict = {} 
paramDict["setC"] = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
paramDict["setGamma"] = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
paramDict["setEpsilon"] = learner.getEpsilons()

foldsSet = numpy.arange(2, 31, 2)
Cvs = numpy.array([1.0])
sampleMethod = Sampling.crossValidation

sampleSize = 100
trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
validX = trainX[trainInds,:]
validY = trainY[trainInds]

"""
for i in range(foldsSet.shape[0]): 
    folds = foldsSet[i]
    
    Cvs = numpy.array([folds-1.0])
    idx = sampleMethod(folds, validY.shape[0])
    svmGridResults = learner.parallelPen(validX, validY, idx, paramDict, Cvs)
    
    
    for result in svmGridResults: 
        learner, trainErrors, currentPenalties = result
        print(numpy.mean(trainErrors), numpy.mean(currentPenalties))
"""

#Figure out why the penalty is increasing 
X = trainX 
y = trainY 

for i in range(foldsSet.shape[0]): 
    folds = foldsSet[i]
    idx = Sampling.crossValidation(folds, validX.shape[0])
    
    penalty = 0
    fullError = 0 
    trainError = 0     
    
    learner.learnModel(validX, validY)
    predY = learner.predict(X)
    predValidY = learner.predict(validX)
    idealPenalty = Evaluator.rootMeanSqError(predY, y) - Evaluator.rootMeanSqError(predValidY, validY)
    
    for trainInds, testInds in idx:
        trainX = validX[trainInds, :]
        trainY = validY[trainInds]
    
        #learner.setGamma(gamma)
        #learner.setC(C)
        learner.learnModel(trainX, trainY)
        predY = learner.predict(validX)
        predTrainY = learner.predict(trainX)
        fullError += Evaluator.rootMeanSqError(predY, validY)
        trainError += Evaluator.rootMeanSqError(predTrainY, trainY)
        penalty += Evaluator.rootMeanSqError(predY, validY) - Evaluator.rootMeanSqError(predTrainY, trainY)
        
    print((folds-1)*fullError/folds, (folds-1)*trainError/folds, (folds-1)*penalty/folds)
    print(idealPenalty)
