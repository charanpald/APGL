"""

Let's compare the SVM with the ego network predictor. 
"""

from apgl.graph import *
from apgl.util import *
from apgl.data import *
from apgl.predictors import *
from apgl.predictors.edge import *
from apgl.kernel import * 
import numpy
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#Load graph and standardise vertices
noise = 0.2
outputDir = PathDefaults.getOutputDir() + "influence/"
filename = outputDir + "SyntheticExamples_n=" + str(noise) + ".spg"
graph = SparseGraph.load(filename)



preprocessor = Standardiser()
vList = graph.getVertexList()
V = vList.getVertices(graph.getAllVertexIds())
V = preprocessor.normaliseArray(V)
vList.setVertices(V)

#graph = graph.subgraph(numpy.arange(200))

print((numpy.trace(numpy.dot(V.T, V))))

logging.info("Number of features " + str(V.shape[1]))
logging.info("IsUndirected: " + str(graph.isUndirected()))

#Split into train and test set. 
allEdges = graph.getAllEdges()
allInds = numpy.random.permutation(allEdges.shape[0])

trainInds = allInds[0:1000]
testInds = allInds[1000:]

trainEdges = allEdges[trainInds, :]
testEdges = allEdges[testInds, :]

trainGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
trainGraph.addEdges(trainEdges, graph.getEdgeValues(trainEdges))

testGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
testGraph.addEdges(testEdges, graph.getEdgeValues(testEdges))

#Do some regression 
svm = LibSVM()
svm.setTermination(0.01)
svm.setSvmType("Epsilon_SVR")
edgeLearner = FlatEdgeLabelPredictor(svm)

lmbda1 = 0.0005
lmbda2 = 0.1
kernel = GaussianKernel()
alterRegressor = PrimalRidgeRegression(lmbda1)
#egoRegressor = KernelRidgeRegression(kernel, lmbda2)
egoRegressor = PrimalRidgeRegression(lmbda2)
edgeLearner2 = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

edgeLearner.learnModel(trainGraph)
edgeLearner2.learnModel(trainGraph)

predY = edgeLearner.predictEdges(trainGraph, trainGraph.getAllEdges())
predY2 = edgeLearner2.predictEdges(trainGraph, trainGraph.getAllEdges())

trainY = trainGraph.getEdgeValues(trainGraph.getAllEdges())

error = Evaluator.rootMeanSqError(trainY, predY)
error2 = Evaluator.rootMeanSqError(trainY, predY2)

print(error)
print(error2)

#Print error of ego and alter predictors
#Centering the array makes things much worse for ego-centric predictor 