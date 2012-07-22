"""
Get some results and compare on a real dataset using SVMs. 
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

gammaInd = 3 
gamma = gammas[gammaInd]
learner.setGamma(gamma)

epsilonInd = 0 
epsilon = epsilons[epsilonInd]
learner.setEpsilon(epsilon)
learner.normModelSelect = True

paramDict = {} 
paramDict["setC"] = Cs 
numParams = Cs.shape[0]

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets: 
    logging.debug("Dataset " + datasetName)
    errors = numpy.zeros(numRealisations)
    sampleMethod = Sampling.crossValidation
    
    alpha = 1.0
    folds = 2
    numRealisations = 5
    numMethods = 6
    sampleSizes = [50, 100, 200]
    
    sampleSizeInd = 2 
    sampleSize = sampleSizes[sampleSizeInd]
    
    #Lets load the learning rates 
    betaFilename = outputDir + datasetName + "Beta.npz"    
    beta = numpy.load(betaFilename)["arr_0"]
    beta = numpy.clip(beta, 0, 1)    

    meanCvGrid = numpy.zeros((numMethods, numParams))
    meanPenalties = numpy.zeros(numParams)
    meanCorrectedPenalties = numpy.zeros(numParams)
    meanBetaPenalties = numpy.zeros(numParams)
    meanIdealPenalities = numpy.zeros(numParams)
    meanAllErrors = numpy.zeros(numParams) 
    meanTrainError = numpy.zeros(numParams)
    meanErrors = numpy.zeros(numMethods)
    
    meanNorms = numpy.zeros(numParams)
    testMeanNorms = numpy.zeros(numParams)
    
    meanSVs = numpy.zeros(numParams)
    testMeanSVs = numpy.zeros(numParams)

    for j in range(numRealisations):
        print("")
        logging.debug("j=" + str(j))
        trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
        logging.debug("Loaded dataset with " + str(trainX.shape) +  " train and " + str(testX.shape) + " test examples")
        
        trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
        trainX = trainX[trainInds,:]
        trainY = trainY[trainInds]
        
        print(trainX.shape)
        
        #logging.debug("Training set size: " + str(trainX.shape))
        methodInd = 0 
        idx = sampleMethod(folds, trainX.shape[0])
        bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
        predY = bestLearner.predict(testX)
        meanCvGrid[methodInd, :] += cvGrid     
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

    
        Cvs = [-5, (folds-1)*alpha, beta[j, sampleSizeInd, :]]    
    
        #Now try penalisation
        methodInd = 1
        resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
        bestLearner, trainErrors, currentPenalties = resultsList[1]
        meanCvGrid[methodInd, :] += trainErrors + currentPenalties
        meanPenalties += currentPenalties
        meanTrainError += trainErrors
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        
        #Corrected penalisation 
        methodInd = 2
        bestLearner, trainErrors, currentPenalties = resultsList[0]
        meanCvGrid[methodInd, :] += trainErrors + currentPenalties
        meanCorrectedPenalties += currentPenalties
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        
        #Learning rate penalisation 
        methodInd = 3
        bestLearner, trainErrors, currentPenalties = resultsList[2]
        meanCvGrid[methodInd, :] += trainErrors + currentPenalties[gammaInd, epsilonInd, :]
        meanBetaPenalties += currentPenalties[gammaInd, epsilonInd, :]
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        
        #Compute ideal penalties and error on training data 
        meanIdealPenalities += learner.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict)

        
        #Compute true error grid 
        methodInd = 4
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, testX, testY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid
        bestLearner = learner.getBestLearner(cvGrid, paramDict, trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

        #Compute true error grid using only training data 
        methodInd = 5
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, trainX, trainY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid
        bestLearner = learner.getBestLearner(cvGrid, paramDict, trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)

    
        #Compute norms 
        tempMeanNorms = numpy.zeros(numParams)
        tempMeanSVs = numpy.zeros(numParams)
        for s, C in enumerate(Cs): 
            for trainInds, testInds in idx: 
                validX = trainX[trainInds, :]
                validY = trainY[trainInds]
                
                learner.setC(C)
                learner.learnModel(validX, validY)
                tempMeanNorms[s] += learner.weightNorm()
                tempMeanSVs[s] += learner.model.support_.shape[0]
            
            learner.learnModel(trainX, trainY)
            testMeanNorms[s] = learner.weightNorm()  
            testMeanSVs[s] += learner.model.support_.shape[0]
            
        tempMeanNorms /= float(folds)
        meanNorms += tempMeanNorms 
        
        tempMeanSVs /= float(folds)
        meanSVs+= tempMeanSVs 
            
    numRealisations = float(numRealisations)
        
    meanCvGrid /=  numRealisations   
    meanPenalties /=  numRealisations   
    meanCorrectedPenalties /= numRealisations 
    meanBetaPenalties /= numRealisations
    meanIdealPenalities /= numRealisations
    meanTrainError /=  numRealisations   
    meanErrors /=  numRealisations 

    meanAllErrors /= numRealisations
    meanNorms /= numRealisations 
    testMeanNorms / numRealisations 
    
    print("\n")
    print("meanErrors=" + str(meanErrors))  
    print(meanIdealPenalities)
    
    #x = numpy.log(Cs)
    #testx = numpy.log(Cs)  
    x = numpy.log(meanNorms)
    testx = numpy.log(testMeanNorms)  
    #x = numpy.log(meanSVs)
    #testx = numpy.log(testMeanSVs)         
    
    plt.figure(figInd)
    plt.plot(x, meanCvGrid[0, :], label="CV")
    plt.plot(x, meanCvGrid[1, :], label="Pen")
    plt.plot(x, meanCvGrid[2, :], label="Corrected Pen")
    plt.plot(x, meanCvGrid[3, :], label="Beta Pen")
    plt.plot(testx, meanCvGrid[4, :], label="Test")
    plt.plot(x, meanCvGrid[5, :], label="Train Error")
    plt.xlabel("log(||w||)")
    plt.ylabel("Error/Penalty")
    plt.legend(loc="lower left")
    #plt.savefig("error_" + datasetName + ".eps")
    figInd += 1
    
    sigma = 5
    idealAlphas = meanIdealPenalities/meanPenalties
    estimatedAlpha = (1-numpy.exp(-sigma*meanTrainError)) + (float(folds)/(folds-1))*numpy.exp(-sigma*meanTrainError)    
    
    plt.figure(figInd)
    plt.plot(x, meanPenalties, label="Penalty")
    plt.plot(x, meanCorrectedPenalties, label="Corrected Penalty")
    plt.plot(x, meanBetaPenalties, label="Beta Penalty")
    plt.plot(testx, meanIdealPenalities, label="Ideal Penalty")
    plt.plot(x, meanTrainError, label="Valid Error")
    plt.plot(x, meanAllErrors, label="Train Error")
    plt.xlabel("log(||w||)")
    plt.ylabel("Error/Penalty")
    plt.legend(loc="center left")
    figInd += 1
    
    
    print("Ideal alphas=" + str(idealAlphas))
    print("Estimated alphas=" + str(estimatedAlpha)) 
    
    plt.show()