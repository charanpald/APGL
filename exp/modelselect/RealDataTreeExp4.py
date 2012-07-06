"""
Try to figure out the learning rate 
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
numpy.random.seed(45)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

sampleSize = 100
foldSizes = numpy.arange(2, 9, 1)

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets:
    logging.debug("Dataset " + datasetName)
    meanErrors = numpy.zeros(foldSizes.shape[0])   
    meanPenalties = numpy.zeros(foldSizes.shape[0])
    meanIdealPenalities = numpy.zeros(foldSizes.shape[0])
    
    k = 0 
    for folds in foldSizes: 
        logging.debug("Folds " + str(folds))
        errors = numpy.zeros(numRealisations)
        
        sampleMethod = Sampling.crossValidation
        
        #Setting maxDepth = 50 and minSplit = 5 doesn't effect results 
        numProcesses = multiprocessing.cpu_count()
        learner = DecisionTreeLearner(criterion="mse", maxDepth=100, minSplit=1, pruneType="CART", processes=numProcesses)
        learner.setChunkSize(3)
        
        paramDict = {} 
        paramDict["setGamma"] = numpy.array([3], dtype=numpy.int)
        numParams = paramDict["setGamma"].shape[0]
        
        alpha = 1.0
        numRealisations = 20
        
        #Cvs = numpy.array([folds-1])*alpha
        Cvs = numpy.array([1])
        
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
    """
    plt.figure(0)
    plt.plot(foldSizes, meanPenalties, label="Penalty")
    plt.plot(foldSizes, meanIdealPenalities, label="Ideal penalty")
    plt.xlabel("Folds")
    plt.ylabel("Penalty")   
    plt.legend()
    """
    tempFoldSizes = numpy.array(foldSizes, numpy.float)    
    
    print(meanIdealPenalities)
    print(meanPenalties)
    print(numpy.log(meanPenalties))
    print(numpy.log((tempFoldSizes-1)/tempFoldSizes*sampleSize))
    print(numpy.log(meanPenalties)+numpy.log(foldSizes))
    
    y = numpy.log(meanPenalties)+numpy.log(foldSizes)    
    x = numpy.log((tempFoldSizes-1)/tempFoldSizes*sampleSize)
    
    print((y[-1]-y[0])/(x[-1]-x[0]))    
    
    plt.figure(1)
    plt.plot(numpy.log((tempFoldSizes-1)/tempFoldSizes*sampleSize), numpy.log(meanPenalties)+numpy.log(foldSizes), label="Penalty")
    plt.xlabel("log(sampleSize*folds)")
    plt.ylabel("log(meanPenalties)")   
    plt.legend()
        
    plt.show()