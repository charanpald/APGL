

"""
Get some results and compare on a real dataset using Penalty Trees, Decision Trees. 
"""
import logging 
import numpy 
import sys 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from apgl.util.Util import Util
from exp.sandbox.predictors.PenaltyDecisionTree import PenaltyDecisionTree

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/benchmark/"

loadMethod = ModelSelectUtils.loadRatschDataset
datasets = ModelSelectUtils.getRatschDatasets(True)
datasetName, numRealisations = datasets[0]

errors = numpy.zeros(numRealisations)
sampleMethod = Sampling.crossValidation

learner = PenaltyDecisionTree()

paramDict = {} 
paramDict["setAlphaThreshold"] = numpy.linspace(0, 0.1, 20)

folds = 5
numRealisations = 5

for j in range(numRealisations): 
    Util.printIteration(j, 1, numRealisations, "Realisation: ")
    trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
    
    idx = sampleMethod(folds, trainX.shape[0])
    bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
    
    predY = bestLearner.predict(testX)
    errors[j] = bestLearner.getMetricMethod()(testY, predY)
    
print(numpy.mean(errors))