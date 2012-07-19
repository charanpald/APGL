"""
Test how the penalty varied for a fixed gamma with the number of examples. 
"""

import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(21)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

sampleSizes = numpy.arange(50, 250, 25)

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets:
    logging.debug("Dataset " + datasetName)
    meanErrors = numpy.zeros(sampleSizes.shape[0])   
    meanPenalties = numpy.zeros(sampleSizes.shape[0])
    meanIdealPenalities = numpy.zeros(sampleSizes.shape[0])
    
    k = 0 
    for sampleSize in sampleSizes: 
        logging.debug("Sample size " + str(sampleSize))
        errors = numpy.zeros(numRealisations)
        
        sampleMethod = Sampling.crossValidation
        
        #Setting maxDepth = 50 and minSplit = 5 doesn't effect results 
        numProcesses = multiprocessing.cpu_count()
        learner = DecisionTreeLearner(criterion="mse", maxDepth=100, minSplit=1, pruneType="CART", processes=numProcesses)
        learner.setChunkSize(3)
        
        paramDict = {} 
        paramDict["setGamma"] = numpy.array([31], dtype=numpy.int)
        numParams = paramDict["setGamma"].shape[0]
        
        alpha = 1.0
        folds = 4
        numRealisations = 10
        
        Cvs = numpy.array([folds-1])*alpha
        
        
        meanAllErrors = numpy.zeros(numParams) 
        meanTrainError = numpy.zeros(numParams)
        

        
        treeSizes = numpy.zeros(numParams)
        treeDepths = numpy.zeros(numParams)
        treeLeaveSizes = numpy.zeros(numParams)
        
        for j in range(numRealisations):
            print("")
            logging.debug("j=" + str(j))
            trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
            logging.debug("Loaded dataset with " + str(trainX.shape) +  " train and " + str(testX.shape) + " test examples")
            
            trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
            trainX = trainX[trainInds,:]
            trainY = trainY[trainInds]
            
            idx = sampleMethod(folds, trainX.shape[0])        
            
            #Now try penalisation
            resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
            bestLearner, trainErrors, currentPenalties = resultsList[0]
            meanPenalties[k] += currentPenalties
            meanTrainError += trainErrors
            predY = bestLearner.predict(testX)
            meanErrors[k] += bestLearner.getMetricMethod()(testY, predY)
    
            
            #Compute ideal penalties and error on training data 
            meanIdealPenalities[k] += learner.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict)
            for i in range(len(paramDict["setGamma"])):
                allError = 0    
                learner.setGamma(paramDict["setGamma"][i])
                for trainInds, testInds in idx: 
                    validX = trainX[trainInds, :]
                    validY = trainY[trainInds]
                    learner.learnModel(validX, validY)
                    predY = learner.predict(trainX)
                    allError += learner.getMetricMethod()(predY, trainY)
                meanAllErrors[i] += allError/float(len(idx))
            
        k+= 1
        
        
    numRealisations = float(numRealisations)
    meanErrors /=  numRealisations 
    meanPenalties /=  numRealisations 
    meanIdealPenalities /=  numRealisations 

    print(meanErrors)
    
    plt.plot(sampleSizes, meanPenalties*numpy.sqrt(sampleSizes), label="Penalty")
    plt.plot(sampleSizes, meanIdealPenalities*numpy.sqrt(sampleSizes), label="Ideal penalty")
    plt.xlabel("Sample sizes")
    plt.ylabel("Penalty")   
    plt.legend()
        
    plt.show()