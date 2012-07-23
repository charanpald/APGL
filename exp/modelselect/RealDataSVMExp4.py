"""
Plot the ideal versus estimated penalty and see where the largest mistakes occur. 
"""
import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from apgl.predictors.LibSVM import LibSVM
from sklearn.linear_model import LinearRegression
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

numCs = Cs.shape[0]
numGammas = gammas.shape[0]
numEpsilons = epsilons.shape[0]

learner.normModelSelect = True

paramDict = {} 
paramDict["setC"] = Cs 
paramDict["setGamma"] = gammas
paramDict["setEpsilon"] = epsilons

#datasets = [datasets[1]]

for datasetName, numRealisations in datasets: 
    logging.debug("Dataset " + datasetName)
    errors = numpy.zeros(numRealisations)
    sampleMethod = Sampling.crossValidation
    
    alpha = 1.0
    folds = 6
    numRealisations = 5
    numMethods = 3
    sampleSizes = [50, 100, 200]
    
    sampleSizeInd = 2 
    sampleSize = sampleSizes[sampleSizeInd]
    
    #Lets load the learning rates 
    betaFilename = outputDir + datasetName + "Beta.npz"    
    beta = numpy.load(betaFilename)["arr_0"]
    beta = numpy.clip(beta, 0, 1)    

    meanPenalties = numpy.zeros((numGammas, numEpsilons, numCs))
    meanBetaPenalties = numpy.zeros((numGammas, numEpsilons, numCs))
    meanIdealPenalities = numpy.zeros((numGammas, numEpsilons, numCs))

    for j in range(numRealisations):
        print("")
        logging.debug("j=" + str(j))
        trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
        logging.debug("Loaded dataset with " + str(trainX.shape) +  " train and " + str(testX.shape) + " test examples")
        
        trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
        trainX = trainX[trainInds,:]
        trainY = trainY[trainInds]
        
        idx = Sampling.crossValidation(folds, trainX.shape[0])

        Cvs = [(folds-1)*alpha, beta[j, sampleSizeInd, :]]    
    
        #Now try penalisation
        methodInd = 0
        resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
        bestLearner, trainErrors, currentPenalties = resultsList[0]
        meanPenalties += currentPenalties
        predY = bestLearner.predict(testX)
                
        #Learning rate penalisation 
        methodInd = 1
        bestLearner, trainErrors, currentPenalties = resultsList[1]
        meanBetaPenalties += currentPenalties
        predY = bestLearner.predict(testX)

        #Compute ideal penalties and error on training data 
        meanIdealPenalities += learner.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict)
            
    numRealisations = float(numRealisations)
        
    meanPenalties /=  numRealisations   
    meanBetaPenalties /= numRealisations
    meanIdealPenalities /= numRealisations

    print("\n")

    approxPenalties = meanPenalties.flatten()    
    betaPenalties = meanBetaPenalties.flatten() 
    idealPenalties = meanIdealPenalities.flatten()
    
    lr = LinearRegression()
    lr.fit(numpy.array([idealPenalties]).T, numpy.array([approxPenalties]).T)
    predApprox = lr.predict(numpy.array([idealPenalties]).T)
    print(lr.coef_)
    
    lr.fit(numpy.array([idealPenalties]).T, numpy.array([betaPenalties]).T)
    predBeta = lr.predict(numpy.array([idealPenalties]).T)    
    print(lr.coef_)
    
    plt.figure(figInd)
    plt.scatter(idealPenalties, approxPenalties, label="Penalty", c="b")
    plt.plot(idealPenalties, predApprox, "b")
    plt.scatter(idealPenalties, betaPenalties, label="Beta", c="r")
    plt.plot(idealPenalties, predBeta, "r")
    plt.xlabel("Ideal Penalty")
    plt.ylabel("Approx Penalty")
    plt.legend(loc="upper left")
    figInd += 1

    plt.show()