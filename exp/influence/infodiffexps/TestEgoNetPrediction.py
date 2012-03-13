"""
Test out the ego-centric predictor on a graph consisting of just the information
decays for positive transmissions. 

"""
import svm

import logging
import random
import sys
import numpy

from apgl.egograph import *
from apgl.io import *
from apgl.util import *
from apgl.graph import *
from apgl.kernel import *
from apgl.data.Standardiser import Standardiser
from apgl.predictors import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
random.seed(21)
numpy.random.seed(21)

#Load up the decay graph
graphFileName = InfoExperiment.getGraphFileName()
decayGraph  = SparseGraph.load(graphFileName)

defaultTol = 10**-5
allEdges = decayGraph.getAllEdges()
edgeDecays = decayGraph.getEdgeValues(allEdges)
#-1 is no transmission, +1 is transmission
binaryEdges = numpy.array(edgeDecays > defaultTol, numpy.int)*2 -1



logging.info("Total number of transmisions: " + str(numpy.sum(binaryEdges[binaryEdges==1])))
logging.warn("About to modify (standardise) the vertices of the graph.")
preprocessor = Standardiser()
V = decayGraph.getVertexList().getVertices(decayGraph.getAllVertexIds())
V = preprocessor.standardiseArray(V)
decayGraph.getVertexList().setVertices(V)

#Take a subgraph
edgeSampleSize = 10000
allEdges = allEdges[0:edgeSampleSize, :]
binaryEdges = binaryEdges[0:edgeSampleSize]

logging.warn("Centering the edge decays for transmissions.")
edgeDecays[binaryEdges==1] = edgeDecays[binaryEdges==1] - numpy.mean(edgeDecays[binaryEdges==1])


#A graph with transmission edges only and the corresponding decays
decayGraph2 = SparseGraph(decayGraph.getVertexList(), False)
decayGraph2.addEdges(allEdges[binaryEdges==1], edgeDecays[binaryEdges==1])
decayGraph2EdgeValues = decayGraph2.getEdgeValues(decayGraph2.getAllEdges())
logging.info("Created transmissions graph with decays " + str(decayGraph2))
logging.info("Min decay " + str(numpy.min(decayGraph2EdgeValues)) + " max decay " + str(numpy.max(decayGraph2EdgeValues)))


#Load params for the decay predictor

lmbda1 = 0.0018
lmbda2 = 0.125
sigma = 0.0015625
kernel = GaussianKernel(sigma)
alterRegressor = PrimalRidgeRegression(lmbda1)
egoRegressor = KernelShiftRegression(kernel, lmbda2)
edgeLearner2 = EgoEdgeLabelPredictor(alterRegressor, egoRegressor) 


"""
svm = LibSVM()
svm.setC(1.0)
svm.setEpsilon(0.0625)
svm.setKernel("gaussian", 0.5)
svm.setTermination(0.01)
svm.setSvmType("Epsilon_SVR")
edgeLearner2 = FlatEdgeLabelPredictor(svm)
"""

edgeLearner2.learnModel(decayGraph2)


#Save the parameters
"""
params = [lmbda2, lmbda1, sigma]
paramFuncs = [egoRegressor.setLambda, alterRegressor.setLambda, kernel.setSigma]
outputDir = PathDefaults.getOutputDir() + "diffusion/"
paramsFile = outputDir + "EgoNetInfoParams.pkl"
edgeLearner2.saveParams(params, paramFuncs, paramsFile)

"""



#Predict edges here:
predDecays = edgeLearner2.predictEdges(decayGraph2, decayGraph2.getAllEdges())
print((Evaluator.rootMeanSqError(decayGraph2.getEdgeValues(decayGraph2.getAllEdges()), numpy.zeros(predDecays.shape))))
print((Evaluator.rootMeanSqError(decayGraph2.getEdgeValues(decayGraph2.getAllEdges()), predDecays)))
print((predDecays[0:20]))
print((decayGraph2.getEdgeValues(decayGraph2.getAllEdges())[0:20]))

#TODO: Model selection for ego-net and SVM ideally 