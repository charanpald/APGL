import multiprocessing
import sys
from apgl.predictors.LibSVM import LibSVM
from apgl.predictors.DecisionTree import DecisionTree
from apgl.predictors.RandomForest import RandomForest
from apgl.util.FileLock import FileLock
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Sampling import Sampling
from apgl.util.Util import Util
from exp.modelselect.ModelSelectUtils import ModelSelectUtils
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import logging
import numpy
import os

"""
Let's run cross validation and model penalisation over some benchmark datasets. 
All datasets are scaled to have unit variance and zero norm. 
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")

def getSetup(learnerName, dataDir, outputDir, numProcesses): 
    
    if learnerName=="SVM":
        learner = LibSVM(kernel='gaussian', type="C_SVC", processes=numProcesses) 
        loadMethod = ModelSelectUtils.loadRatschDataset
        dataDir += "benchmark/"
        outputDir += "classification/" + learnerName + "/"
        
        paramDict = {} 
        paramDict["setC"] = learner.getCs()
        paramDict["setGamma"] = learner.getGammas()  
    elif learnerName=="SVR":
        learner = LibSVM(kernel='gaussian', type="Epsilon_SVR", processes=numProcesses) 
        loadMethod = ModelSelectUtils.loadRegressDataset
        dataDir += "regression/"
        outputDir += "regression/" + learnerName + "/"

        paramDict = {} 
        paramDict["setC"] = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
        paramDict["setGamma"] = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
        paramDict["setEpsilon"] = learner.getEpsilons()
    elif learnerName=="DTC": 
        learner = DecisionTree()
        loadMethod = ModelSelectUtils.loadRatschDataset
        dataDir += "benchmark/"
        outputDir += "classification/" + learnerName + "/"

        paramDict = {} 
        paramDict["setMaxDepth"] = numpy.arange(1, 31, 2)
        paramDict["setMinSplit"] = 2**numpy.arange(1, 7, dtype=numpy.int) 
    elif learnerName=="RFR": 
        learner = RandomForest(criterion="mse", type="reg")
        loadMethod = ModelSelectUtils.loadRegressDataset
        dataDir += "regression/"
        outputDir += "regression/" + learnerName + "/"

        paramDict = {} 
        paramDict["setNumTrees"] = 2**numpy.arange(3, 9, 2, dtype=numpy.int) 
        paramDict["setMinSplit"] = 2**numpy.arange(2, 6, dtype=numpy.int) 
    elif learnerName=="CART": 
        learner = DecisionTreeLearner(criterion="mse", maxDepth=30, minSplit=1, pruneType="CART", processes=numProcesses)
        learner.setChunkSize(2)
        loadMethod = ModelSelectUtils.loadRegressDataset
        dataDir += "regression/"
        outputDir += "regression/" + learnerName + "/"

        paramDict = {} 
        paramDict["setGamma"] =  numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)
    else: 
        raise ValueError("Unknown learnerName: " + learnerName)
                
    return learner, loadMethod, dataDir, outputDir, paramDict 

def runBenchmarkExp(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix, learnerName, betaNameSuffix):
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/"
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/"

    learner, loadMethod, dataDir, outputDir, paramDict = getSetup(learnerName, dataDir, outputDir, numProcesses)
    numParams = len(paramDict.keys())
    numMethods = 1 + (cvScalings.shape[0] + 1)
    
    runCv = True
    runVfpen = True

    for i in range(len(datasetNames)):
        datasetName = datasetNames[i][0]
        numRealisations = datasetNames[i][1]
        logging.debug("Learning using dataset " + datasetName)
        
        #Load learning rates for penalisation 
        betafileName = outputDir + datasetNames[i][0] + betaNameSuffix + ".npz"
        betaGrids = numpy.load(betafileName)["arr_0"]
        betaGrids = numpy.clip(betaGrids, 0, 1)    

        for s in range(len(sampleMethods)):
            sampleMethod = sampleMethods[s][1]
            outfileName = outputDir + datasetName + sampleMethods[s][0] + fileNameSuffix

            fileLock = FileLock(outfileName + ".npz")
            if not fileLock.isLocked() and not fileLock.fileExists():
                fileLock.lock()
                errors = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods))
                params = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods, numParams))
                
                errorShape = [numRealisations, len(sampleSizes), foldsSet.shape[0] ,numMethods]
                errorShape.extend(list(learner.gridShape(paramDict))) 
                errorShape = tuple(errorShape)
                
                gridShape = [numRealisations, len(sampleSizes), foldsSet.shape[0] ,numMethods]
                gridShape.extend(list(learner.gridShape(paramDict)))   
                gridShape = tuple(gridShape)
                
                idealErrorShape = [numRealisations, len(sampleSizes)]
                idealErrorShape.extend(list(learner.gridShape(paramDict)))   
                idealErrorShape = tuple(idealErrorShape)
                
                errorGrids = numpy.zeros(errorShape)
                approxGrids = numpy.zeros(errorShape)

                for j in range(numRealisations):
                    Util.printIteration(j, 1, numRealisations, "Realisation: ")
                    trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
                  
                    for k in range(sampleSizes.shape[0]):
                        sampleSize = sampleSizes[k]
                        for m in range(foldsSet.shape[0]):
                            if foldsSet[m] < sampleSize: 
                                folds = foldsSet[m]
                            else: 
                                folds = sampleSize 
                            logging.debug("Using sample size " + str(sampleSize) + " and " + str(folds) + " folds")
                            numpy.random.seed(21)
                            trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
                            validX = trainX[trainInds,:]
                            validY = trainY[trainInds]

                            #Cross validation
                            if runCv:
                                logging.debug("Running simple sampling using " + str(sampleMethod))
                                methodInd = 0
                                idx = sampleMethod(folds, validY.shape[0])
                                bestLearner, cvGrid = learner.parallelModelSelect(validX, validY, idx, paramDict)
                                predY = bestLearner.predict(testX)
                                errors[j, k, m, methodInd] = bestLearner.getMetricMethod()(testY, predY)
                                params[j, k, m, methodInd, :] = bestLearner.getParamsArray(paramDict)
                                errorGrids[j, k, m, methodInd, :] = cvGrid

                            #v fold penalisation
                            if runVfpen:
                                logging.debug("Running penalisation using " + str(sampleMethod))
                                #Corrected penalisation give by using learning rate 

                                tempCvScalings = list(cvScalings * (folds-1))
                                tempCvScalings.insert(0, betaGrids[j, k, :]) 
                                
                                idx = sampleMethod(folds, validY.shape[0])
                                learnerGridResults = learner.parallelPen(validX, validY, idx, paramDict, tempCvScalings)

                                for n in range(len(tempCvScalings)):
                                    bestLearner, trainErrors, approxGrid = learnerGridResults[n]
                                    predY = bestLearner.predict(testX)
                                    methodInd = n + 1
                                    errors[j, k, m, methodInd] = bestLearner.getMetricMethod()(testY, predY)
                                    params[j, k, m, methodInd, :] = bestLearner.getParamsArray(paramDict)
                                    errorGrids[j, k, m, methodInd, :] = trainErrors + approxGrid
                                    approxGrids[j, k, m, methodInd, :] = approxGrid
                        
                meanErrors = numpy.mean(errors, 0)
                print(meanErrors)

                meanParams = numpy.mean(params, 0)
                print(meanParams)

                #When using CART trees the penalty can be inf in which case std is undefined
                #In this case we set to zero any infinite values 
                meanErrorGrids = numpy.mean(errorGrids, 0)
                try: 
                    stdErrorGrids = numpy.std(errorGrids, 0)
                except FloatingPointError:
                    errorGrids[numpy.isinf(errorGrids)] = 0 
                    stdErrorGrids = numpy.std(errorGrids, 0)
            
                meanApproxGrids = numpy.mean(approxGrids, 0) 
                try: 
                    stdApproxGrids = numpy.std(approxGrids, 0)
                except FloatingPointError:
                    approxGrids[numpy.isinf(approxGrids)] = 0 
                    stdApproxGrids = numpy.std(approxGrids, 0)

                numpy.savez(outfileName, errors, params, meanErrorGrids, stdErrorGrids, meanApproxGrids, stdApproxGrids)
                logging.debug("Saved results as file " + outfileName + ".npz")
                fileLock.unlock()
            else:
                logging.debug("Results already computed")
            
    logging.debug("All done!")


def findErrorGrid(datasetNames, numProcesses, fileNameSuffix, learnerName, sampleSizes): 
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/"
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/"

    learner, loadMethod, dataDir, outputDir, paramDict = getSetup(learnerName, dataDir, outputDir, numProcesses)
    
    numParams = len(paramDict.keys())    
    
    runIdeal = True 
    runTest = True 
    
    for i in range(len(datasetNames)):
        logging.debug("Learning using dataset " + datasetNames[i][0])
        outfileName = outputDir + datasetNames[i][0] + fileNameSuffix
    
        fileLock = FileLock(outfileName + ".npz")
        if not fileLock.isLocked() and not fileLock.fileExists():
            fileLock.lock()
            
            numRealisations = datasetNames[i][1]            
            
            gridShape = [numRealisations, sampleSizes.shape[0]]
            gridShape.extend(list(learner.gridShape(paramDict)))   
            gridShape = tuple(gridShape)   
            
            idealPenGrids = numpy.zeros(gridShape)
            idealErrorGrids = numpy.zeros(gridShape)
            idealErrors = numpy.zeros((numRealisations, sampleSizes.shape[0]))
                
            params = numpy.zeros((numRealisations, len(sampleSizes), numParams))    
            
            for k in range(sampleSizes.shape[0]):
                sampleSize = sampleSizes[k]
                
                logging.debug("Using sample size " + str(sampleSize))
                for j in range(numRealisations):
                        Util.printIteration(j, 1, numRealisations, "Realisation: ")
                        trainX, trainY, testX, testY = loadMethod(dataDir, datasetNames[i][0], j)
                        
                        numpy.random.seed(21)
                        trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
                        validX = trainX[trainInds,:]
                        validY = trainY[trainInds]
                        
                        #Find ideal penalties
                        if runIdeal:
                            logging.debug("Finding ideal grid of penalties")
                            idealPenGrids[j, k, :] = learner.parallelPenaltyGrid(validX, validY, testX, testY, paramDict)                        
                        
                        #Find ideal model using the test set 
                        if runTest: 
                            logging.debug("Running test set sampling") 
                            cvGrid = learner.parallelSplitGrid(validX, validY, testX, testY, paramDict)
                            bestLearner = learner.getBestLearner(cvGrid, paramDict, validX, validY)
                            predY = bestLearner.predict(testX)
                            idealErrors[j, k] = bestLearner.getMetricMethod()(testY, predY)
                            params[j, k, :] = bestLearner.getParamsArray(paramDict)
                            idealErrorGrids[j, k, :] = cvGrid
            
            meanIdealPenGrids = idealPenGrids.mean(0)
            stdIdealPenGrids = idealPenGrids.std(0)
            
            meanIdealErrorGrids = idealErrorGrids.mean(0)
            stdIdealErrorGrids = idealErrorGrids.std(0)
            
            numpy.savez(outfileName, idealErrors, params, meanIdealErrorGrids, stdIdealErrorGrids, meanIdealPenGrids, stdIdealPenGrids)
            logging.debug("Saved results as file " + outfileName + ".npz")
            fileLock.unlock()
        else:
            logging.debug("Results already computed")


def computeLearningRates(datasetNames, numProcesses, fileNameSuffix, learnerName, sampleSizes, foldsSet): 
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/"
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/"

    learner, loadMethod, dataDir, outputDir, paramDict = getSetup(learnerName, dataDir, outputDir, numProcesses)
    
    for i in range(len(datasetNames)):
        logging.debug("Learning using dataset " + datasetNames[i][0])
        outfileName = outputDir + datasetNames[i][0] + fileNameSuffix

        fileLock = FileLock(outfileName + ".npz")
        if not fileLock.isLocked() and not fileLock.fileExists():
            fileLock.lock()
            
            numRealisations = datasetNames[i][1]  
            gridShape = [numRealisations, sampleSizes.shape[0]]
            gridShape.extend(list(learner.gridShape(paramDict)))   
            gridShape = tuple(gridShape)            
            
            betaGrids = numpy.zeros(gridShape) 
            
            for k in range(sampleSizes.shape[0]):
                sampleSize = sampleSizes[k]
                
                logging.debug("Using sample size " + str(sampleSize))
                for j in range(numRealisations):
                        Util.printIteration(j, 1, numRealisations, "Realisation: ")
                        trainX, trainY, testX, testY = loadMethod(dataDir, datasetNames[i][0], j)
                        
                        numpy.random.seed(21)
                        trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
                        validX = trainX[trainInds,:]
                        validY = trainY[trainInds]
                        
                        betaGrids[j, k, :] = learner.learningRate(validX, validY, foldsSet, paramDict)
            
            numpy.savez(outfileName, betaGrids)
            logging.debug("Saved results as file " + outfileName + ".npz")
            fileLock.unlock()
                        
                        
def shuffleSplit66(repetitions, numExamples):
    """
    Take two thirds of the examples to train, and the rest to test 
    """
    return Sampling.shuffleSplit(repetitions, numExamples, 2.0/3.0)

def shuffleSplit90(repetitions, numExamples):
    """
    Take two thirds of the examples to train, and the rest to test

    """
    return Sampling.shuffleSplit(repetitions, numExamples, 0.9)

def repCrossValidation3(folds, numExamples): 
    return Sampling.repCrossValidation(folds, numExamples, repetitions=3)

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()


#sampleMethods = [("CV", Sampling.crossValidation), ("SS", Sampling.shuffleSplit), ("SS66", shuffleSplit66), ("SS90", shuffleSplit90), ("RCV", repCrossValidation3)]
cvScalings = numpy.arange(0.6, 1.61, 0.2)
betaFoldsSet = numpy.arange(2, 13, 1)

betaNameSuffix = "Beta"
fileNameSuffix = "Results" 
paramGridSuffix = "GridResults" 
betaNameExtSuffix = "BetaExt"
extFileNameSuffix = "ResultsExt"
extParamGridSuffix = "GridResultsExt" 

regressiondatasetNames = ModelSelectUtils.getRegressionDatasets(True)

logging.debug("Using " + str(numProcesses) + " processes")
logging.debug("Process id: " + str(os.getpid()))

runSVR = False 
runCART = True 

if runSVR: 
    learnerName = "SVR"

    sampleMethods = [("CV", Sampling.crossValidation)]
    sampleSizes = numpy.array([50, 100, 200])
    foldsSet = numpy.arange(2, 13, 2)      
    
    computeLearningRates(regressiondatasetNames, numProcesses, betaNameSuffix, learnerName, sampleSizes, betaFoldsSet)
    findErrorGrid(regressiondatasetNames, numProcesses, paramGridSuffix, learnerName, sampleSizes)
    runBenchmarkExp(regressiondatasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix, learnerName, betaNameSuffix)
    
    extSampleMethods = [("CV", Sampling.crossValidation)]
    extSampleSizes = numpy.array([25, 50, 100])
    extFoldsSet = numpy.arange(10, 51, 10)

    computeLearningRates(regressiondatasetNames, numProcesses, betaNameExtSuffix, learnerName, extSampleSizes, betaFoldsSet)
    findErrorGrid(regressiondatasetNames, numProcesses, extParamGridSuffix, learnerName, extSampleSizes)
    runBenchmarkExp(regressiondatasetNames, extSampleSizes, extFoldsSet, cvScalings, extSampleMethods, numProcesses, extFileNameSuffix, learnerName, betaNameExtSuffix)

if runCART: 
    learnerName = "CART"
    
    sampleMethods = [("CV", Sampling.crossValidation)]
    sampleSizes = numpy.array([50, 100, 200])
    foldsSet = numpy.arange(2, 13, 2)    
    
    computeLearningRates(regressiondatasetNames, numProcesses, betaNameSuffix, learnerName, sampleSizes, betaFoldsSet)
    findErrorGrid(regressiondatasetNames, numProcesses, paramGridSuffix, learnerName, sampleSizes)
    runBenchmarkExp(regressiondatasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix, learnerName, betaNameSuffix)
    
    extSampleMethods = [("CV", Sampling.crossValidation)]
    extSampleSizes = numpy.array([500, 1000])
    extFoldsSet = numpy.arange(2, 13, 2)
    
    computeLearningRates(regressiondatasetNames, numProcesses, betaNameExtSuffix, learnerName, extSampleSizes, betaFoldsSet)
    findErrorGrid(regressiondatasetNames, numProcesses, extParamGridSuffix, learnerName, extSampleSizes)
    runBenchmarkExp(regressiondatasetNames, extSampleSizes, extFoldsSet, cvScalings, extSampleMethods, numProcesses, extFileNameSuffix, learnerName, betaNameExtSuffix)
    