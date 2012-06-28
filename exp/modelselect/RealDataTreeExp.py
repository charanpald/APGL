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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(21)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

datasets = [datasets[5]]

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
    paramDict["setGamma"] = numpy.array(numpy.round(2**numpy.arange(1, 10, 0.5)-1), dtype=numpy.int)
    numParams = paramDict["setGamma"].shape[0]
    
    alpha = 1.0
    folds = 5
    numRealisations = 10
    numMethods = 4
    sampleSize = 100 
    Cvs = numpy.array([folds-1])*alpha
    
    meanCvGrid = numpy.zeros((numMethods, numParams))
    meanPenalties = numpy.zeros(numParams)
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
        
        idx = sampleMethod(folds, trainX.shape[0])
        bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
        predY = bestLearner.predict(testX)
        meanCvGrid[0, :] += cvGrid     
        meanErrors[0] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[0] += bestLearner.tree.depth()
        meanSizes[0] += bestLearner.tree.getNumVertices()
    
        #Now try penalisation
        resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
        bestLearner, trainErrors, currentPenalties = resultsList[0]
        meanCvGrid[1, :] += trainErrors + currentPenalties
        meanPenalties += currentPenalties
        meanTrainError += trainErrors
        predY = bestLearner.predict(testX)
        meanErrors[1] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[1] += bestLearner.tree.depth()
        meanSizes[1] += bestLearner.tree.getNumVertices()
        
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
        cvGrid  = learner.parallelSplitGrid(trainX, trainY, testX, testY, paramDict)    
        meanCvGrid[2, :] += cvGrid
        bestLearner.setGamma(paramDict["setGamma"][numpy.argmin(cvGrid)])
        bestLearner.learnModel(trainX, trainY)
        predY = bestLearner.predict(testX)
        meanErrors[2] += bestLearner.getMetricMethod()(testY, predY)
        meanDepths[2] += bestLearner.tree.depth()
        meanSizes[2] += bestLearner.tree.getNumVertices()
        
        #Compute true error grid using only training data 
        methodInd = 3
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
    print("treeDepths=" + str(treeDepths))
    print("treeLeaveSizes=" + str(treeLeaveSizes))
    
    plt.figure(figInd)
    plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[0, :], label="CV")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[1, :], label="Pen")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[2, :], label="Test")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[3, :], label="Train Error")
    plt.xlabel("log(gamma)")
    plt.ylabel("Error/Penalty")
    plt.legend()
    #plt.savefig("error_" + datasetName + ".eps")
    figInd += 1
    
    plt.figure(figInd)
    plt.plot(numpy.log2(paramDict["setGamma"]), meanPenalties, label="Penalty")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanIdealPenalities, label="Ideal Penalty")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanTrainError, label="Valid Error")
    plt.plot(numpy.log2(paramDict["setGamma"]), meanAllErrors, label="Train Error")
    plt.xlabel("log(gamma)")
    plt.ylabel("Error/Penalty")
    plt.legend()
    figInd += 1
    
    plt.show()