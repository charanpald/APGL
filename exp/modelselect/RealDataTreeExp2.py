import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import matplotlib.pyplot as plt 

"""
Learn more about the cases where penalisation fails.
"""



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(21)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)
datasetName, numRealisations = datasets[2]

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

#Test penalty and cross validation error is correct 
trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, 1)
print(trainX.shape, testX.shape)

sampleSize = 100
trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
trainX = trainX[trainInds,:]
trainY = trainY[trainInds]

learner.setGamma(500)

#Look at test examples in tree 
learner.learnModel(trainX, trainY)
learner.predict(trainX)

leafSizes = [] 
for leaf in learner.tree.leaves(): 
    leafSizes.append(learner.tree.getVertex(leaf).getTestInds().shape[0])

leafSizes = numpy.array(leafSizes)
leafSizes = numpy.sort(leafSizes)/float(leafSizes.sum())
hist = numpy.histogram(leafSizes)

print(learner.tree.getNumVertices())

plt.figure(0)
print(hist)
plt.bar(hist[0], hist[1][0:-1])

#Now prune tree
learner.setGamma(500)
learner.learnModel(trainX, trainY)
learner.predict(testX)

leafSizes = [] 
for leaf in learner.tree.leaves(): 
    leafSizes.append(learner.tree.getVertex(leaf).getTestInds().shape[0])

leafSizes = numpy.array(leafSizes)
leafSizes = numpy.sort(leafSizes)/float(leafSizes.sum())
hist = numpy.histogram(leafSizes)

print(learner.tree.getNumVertices())

plt.figure(1)
plt.bar(hist[0], hist[1][0:-1])

plt.show()
