"""
Get some results and compare on a real dataset using Decision Trees. 
"""
import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from apgl.predictors.LibSVM import LibSVM
import matplotlib.pyplot as plt 
from apgl.util import Util 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(21)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/SVR/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

numProcesses = multiprocessing.cpu_count()
learner = LibSVM(kernel="rbf", processes=numProcesses, type="Epsilon_SVR")
learner.setChunkSize(3)

Cs = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
gammas = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
epsilons = learner.getEpsilons()

numCs = Cs.shape[0]
numGammas = gammas.shape[0]

paramDict = {} 
paramDict["setC"] = Cs 
paramDict["setGamma"] = gammas
paramDict["setEpsilon"] = epsilons 

print(learner)

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets:
    logging.debug("Dataset " + datasetName)
    
    errors = numpy.zeros(numRealisations)
    sampleMethod = Sampling.crossValidation
    
    alpha = 1.0
    folds = 4
    numRealisations = 5
    numMethods = 6
    sampleSizes = [50, 100, 200]
    
    sampleSizeInd = 2 
    sampleSize = sampleSizes[sampleSizeInd]
    
    #Lets load the learning rates 
    betaFilename = outputDir + datasetName + "Beta.npz"    
    beta = numpy.load(betaFilename)["arr_0"]
    beta = numpy.clip(beta, 0, 1)    

    meanCvGrid = numpy.zeros((numMethods, numGammas, numCs))
    meanPenalties = numpy.zeros((numGammas, numCs))
    meanCorrectedPenalties = numpy.zeros((numGammas, numCs))
    meanBetaPenalties = numpy.zeros((numGammas, numCs))
    meanIdealPenalities = numpy.zeros((numGammas, numCs))
    meanTrainError = numpy.zeros((numGammas, numCs))
    meanErrors = numpy.zeros((numGammas, numCs))

    
    for j in range(numRealisations):
        logging.debug("j=" + str(j))
        trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
        logging.debug("Loaded dataset with " + str(trainX.shape) +  " train and " + str(testX.shape) + " test examples")
        
        trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
        trainX = trainX[trainInds,:]
        trainY = trainY[trainInds]
        
        #logging.debug("Training set size: " + str(trainX.shape))
        methodInd = 0 
        idx = sampleMethod(folds, trainX.shape[0])
        bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
        predY = bestLearner.predict(testX)
        meanCvGrid[methodInd, :] += cvGrid.min(1)     
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        Cvs = [-5, (folds-1)*alpha, beta[j, sampleSizeInd, :]]    
    
        #Now try penalisation
        methodInd = 1
        resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
        bestLearner, trainErrors, currentPenalties = resultsList[1]
        meanCvGrid[methodInd, :] += (trainErrors + currentPenalties).min(1)
        meanPenalties += currentPenalties.min(1)
        meanTrainError += trainErrors.min(1)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        
        #Corrected penalisation 
        methodInd = 2
        bestLearner, trainErrors, currentPenalties = resultsList[0]
        meanCvGrid[methodInd, :] += (trainErrors + currentPenalties).min(1)
        meanCorrectedPenalties += currentPenalties.min(1)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        
        #Learning rate penalisation 
        methodInd = 3
        bestLearner, trainErrors, currentPenalties = resultsList[2]
        meanCvGrid[methodInd, :] += (trainErrors + currentPenalties).min(1)
        meanBetaPenalties += currentPenalties.min(1)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        
        #Compute ideal penalties and error on training data 
        meanIdealPenalities += learner.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict).min(1)

        
        #Compute true error grid 
        methodInd = 4
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, testX, testY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid.min(1)
        bestLearner = learner.getBestLearner(cvGrid, paramDict, trainX, trainY)
        bestLearner.learnModel(trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        
        #Compute true error grid using only training data 
        methodInd = 5
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, trainX, trainY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid.min(1)
        bestLearner = learner.getBestLearner(cvGrid, paramDict, trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        
        
    numRealisations = float(numRealisations)
        
    meanCvGrid /=  numRealisations   
    meanPenalties /=  numRealisations   
    meanCorrectedPenalties /= numRealisations 
    meanBetaPenalties /= numRealisations
    meanIdealPenalities /= numRealisations
    meanTrainError /=  numRealisations   
    meanErrors /=  numRealisations 
    
    print("\n")

    labels = ["CV", "Pen", "Corrected Pen", "Beta Pen", "Test", "Train Error"]

    for figInd in range(numMethods):   
        plt.figure(figInd)
        plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanCvGrid[figInd, :])
        plt.xlabel("log(C)")
        plt.ylabel("log(gamma)")
        plt.title(labels[figInd])
        plt.colorbar()

    
    figInd = numMethods
    
    plt.figure(figInd)
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanPenalties)
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()
    plt.title("Penalty")
    figInd += 1 
    
    plt.figure(figInd)
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanCorrectedPenalties)
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()
    plt.title("Corrected Penalty")
    figInd += 1 
    
    plt.figure(figInd)
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanBetaPenalties)
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()
    plt.title("Beta Penalty")
    figInd += 1 
    
    
    plt.figure(figInd)
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanIdealPenalities)
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()
    plt.title("Ideal Penalty")
    figInd += 1 
    
    plt.figure(figInd)
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), meanTrainError)
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.colorbar()
    plt.title("Valid Error")
    figInd += 1 
    
    
    plt.show()