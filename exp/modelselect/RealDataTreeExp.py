"""
Get some results and compare on a real dataset using Penalty Trees, Decision Trees. 
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

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)
datasetName, numRealisations = datasets[0]

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

alpha = 1
folds = 5
numRealisations = 10
numMethods = 3
sampleSize = 100 
Cvs = numpy.array([folds-1])*alpha

meanCvGrid = numpy.zeros((numMethods, numParams))
meanPenalties = numpy.zeros(numParams)
meanTrainError = numpy.zeros(numParams)
meanErrors = numpy.zeros(numMethods)
meanDepths = numpy.zeros(numMethods)
meanSizes = numpy.zeros(numMethods)

treeSizes = numpy.zeros(numParams)
treeDepths = numpy.zeros(numParams)

for j in range(numRealisations):
    print("")
    logging.debug("j=" + str(j))
    trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
    logging.debug("Loaded dataset")
    
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
    
    #Compute true error grid 
    cvGrid  = learner.parallelSplitGrid(trainX, trainY, testX, testY, paramDict)    
    meanCvGrid[2, :] += cvGrid
    bestLearner.setGamma(paramDict["setGamma"][numpy.argmin(cvGrid)])
    bestLearner.learnModel(trainX, trainY)
    predY = bestLearner.predict(testX)
    meanErrors[2] += bestLearner.getMetricMethod()(testY, predY)
    meanDepths[2] += bestLearner.tree.depth()
    meanSizes[2] += bestLearner.tree.getNumVertices()
    
    #Compute tree sizes 
    i = 0 
    for gamma in paramDict["setGamma"]: 
        learner.setGamma(gamma)
        learner.learnModel(trainX, trainY)
        treeSizes[i] = learner.tree.getNumVertices()
        treeDepths[i] = learner.tree.depth()
        i +=1 
    
meanCvGrid /=  numRealisations   
meanPenalties /=  numRealisations   
meanTrainError /=  numRealisations   
meanErrors /=  numRealisations 
meanDepths /= numRealisations
meanSizes /= numRealisations

print("\n")
print("meanErrors=" + str(meanErrors))
print("meanDepths=" + str(meanDepths))
print("meanSizes=" + str(meanSizes))

print("treeSizes=" + str(treeSizes))
print("treeDepths=" + str(treeDepths))

plt.figure(0)
plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[0, :], label="CV")
plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[1, :], label="Pen")
plt.plot(numpy.log2(paramDict["setGamma"]), meanCvGrid[2, :], label="Test")
plt.xlabel("log(gamma)")
plt.ylabel("Error/Penalty")
plt.legend()

plt.figure(1)
plt.plot(numpy.log2(paramDict["setGamma"]), meanPenalties, label="Penalty")
plt.plot(numpy.log2(paramDict["setGamma"]), meanTrainError, label="Train Error")
plt.xlabel("log(gamma)")
plt.ylabel("Error/Penalty")
plt.legend()
    
plt.show()