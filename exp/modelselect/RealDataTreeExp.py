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
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import matplotlib.pyplot as plt 
from apgl.util import Util 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(21)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

gammas = numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets:
    #Comp-activ and concrete are bad cases 
    #pumadyn-32nh is good one 
    
    logging.debug("Dataset " + datasetName)
    
    errors = numpy.zeros(numRealisations)
    
    def repCrossValidation(folds, numExamples): 
        return Sampling.repCrossValidation(folds, numExamples, repetitions=3)
    
    sampleMethod = Sampling.crossValidation
    
    #Setting maxDepth = 50 and minSplit = 5 doesn't effect results 
    numProcesses = multiprocessing.cpu_count()
    learner = DecisionTreeLearner(criterion="mse", maxDepth=100, minSplit=1, pruneType="CART", processes=numProcesses)
    learner.setChunkSize(3)
    
    paramDict = {} 
    paramDict["setGamma"] = gammas
    numParams = paramDict["setGamma"].shape[0]
    
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
    meanDepths = numpy.zeros(numMethods)
    meanSizes = numpy.zeros(numMethods)
    
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
        
        #logging.debug("Training set size: " + str(trainX.shape))
        methodInd = 0 
        idx = sampleMethod(folds, trainX.shape[0])
        bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
        predY = bestLearner.predict(testX)
        meanCvGrid[methodInd, :] += cvGrid     
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
    
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
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
        
        #Corrected penalisation 
        methodInd = 2
        bestLearner, trainErrors, currentPenalties = resultsList[0]
        meanCvGrid[methodInd, :] += trainErrors + currentPenalties
        meanCorrectedPenalties += currentPenalties
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
        
        #Learning rate penalisation 
        methodInd = 3
        bestLearner, trainErrors, currentPenalties = resultsList[2]
        meanCvGrid[methodInd, :] += trainErrors + currentPenalties
        meanBetaPenalties += currentPenalties
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
        
        #Compute ideal penalties and error on training data 
        meanIdealPenalities += learner.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict)
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
        
        #Compute true error grid 
        methodInd = 4
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, testX, testY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid
        bestLearner.setGamma(paramDict["setGamma"][numpy.argmin(cvGrid)])
        bestLearner.learnModel(trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
        
        #Compute true error grid using only training data 
        methodInd = 5
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, trainX, trainY, paramDict)    
        meanCvGrid[methodInd, :] += cvGrid
        bestLearner.setGamma(paramDict["setGamma"][numpy.argmin(cvGrid)])
        bestLearner.learnModel(trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[methodInd] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[methodInd] += bestLearner.tree.depth()
        meanSizes[methodInd] += bestLearner.tree.getNumVertices()
        
        #Compute tree properties 
        i = 0 
        for gamma in paramDict["setGamma"]: 
            learner.setGamma(gamma)
            learner.learnModel(trainX, trainY)
            treeSizes[i] = learner.tree.getNumVertices()
            treeDepths[i] = learner.tree.depth()
            
            tempMean = 0 
            for leaf in learner.tree.leaves(): 
                tempMean += learner.tree.getVertex(leaf).getTrainInds().shape[0]
    
            tempMean /= float(len(learner.tree.leaves()))
            treeLeaveSizes[i] += tempMean 
            
            i +=1 
        
    numRealisations = float(numRealisations)
        
    meanCvGrid /=  numRealisations   
    meanPenalties /=  numRealisations   
    meanCorrectedPenalties /= numRealisations 
    meanBetaPenalties /= numRealisations
    meanIdealPenalities /= numRealisations
    meanTrainError /=  numRealisations   
    meanErrors /=  numRealisations 
    meanDepths /= numRealisations
    meanSizes /= numRealisations
    treeLeaveSizes /= numRealisations
    meanAllErrors /= numRealisations
    
    print("\n")
    print("meanErrors=" + str(meanErrors))
    print("meanDepths=" + str(meanDepths))
    print("meanSizes=" + str(meanSizes))
    
    print("Test error" + str(meanCvGrid[2, :]))
    print("treeSizes=" + str(treeSizes))
    print("gammas=" + str(paramDict["setGamma"]))
    print("treeDepths=" + str(treeDepths))
    print("treeLeaveSizes=" + str(treeLeaveSizes))
    
    gammas = paramDict["setGamma"]
    inds = numpy.arange(gammas.shape[0])[meanCvGrid[1, :] < float("inf")]   
    
    plt.figure(figInd)
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[0, inds], label="CV")
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[1, inds], label="Pen")
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[2, inds], label="Corrected Pen")
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[3, inds], label="Beta Pen")
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[4, inds], label="Test")
    plt.plot(numpy.log2(gammas[inds]), meanCvGrid[5, inds], label="Train Error")
    plt.xlabel("log(gamma)")
    plt.ylabel("Error/Penalty")
    plt.legend(loc="lower left")
    #plt.savefig("error_" + datasetName + ".eps")
    figInd += 1
    
    sigma = 5
    idealAlphas = meanIdealPenalities/meanPenalties
    estimatedAlpha = (1-numpy.exp(-sigma*meanTrainError)) + (float(folds)/(folds-1))*numpy.exp(-sigma*meanTrainError)    
    
    plt.figure(figInd)
    plt.plot(numpy.log2(treeSizes), meanPenalties, label="Penalty")
    plt.plot(numpy.log2(treeSizes), meanCorrectedPenalties, label="Corrected Penalty")
    plt.plot(numpy.log2(treeSizes), meanBetaPenalties, label="Beta Penalty")
    plt.plot(numpy.log2(treeSizes), meanIdealPenalities, label="Ideal Penalty")
    plt.plot(numpy.log2(treeSizes), meanTrainError, label="Valid Error")
    plt.plot(numpy.log2(treeSizes), meanAllErrors, label="Train Error")
    plt.xlabel("log(treeSize)")
    plt.ylabel("Error/Penalty")
    plt.legend(loc="center left")
    figInd += 1
    
    
    print("Ideal alphas=" + str(idealAlphas))
    print("Estimated alphas=" + str(estimatedAlpha)) 
    
    plt.show()