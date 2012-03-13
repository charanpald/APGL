"""
Find SVM parameters using cross validation and then evaluate the error, and
save the results.
"""
from apgl.data import *
from apgl.util import *
from apgl.predictors import *
from apgl.predictors.edge import *
from apgl.graph import *
from apgl.kernel import *
import logging
import sys
import numpy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

noises = [0.0, 0.05, 0.1, 0.15, 0.2]
sampleSize = 10000
#sampleSize = 1000
folds = 5
kernel = "linear"
lmbdas1 = 2.0**numpy.arange(-7, 0)
lmbdas2 = 2.0**numpy.arange(-7, 1)
errorFunc = Evaluator.rootMeanSqError

lmbda = 0.1
kernel = LinearKernel()
alterRegressor = PrimalRidgeRegression(lmbda)
egoRegressor = KernelShiftRegression(kernel, lmbda)

edgeLearner = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

paramList = []
paramFuncs = [alterRegressor.setLambda, egoRegressor.setLambda]

for lmbda1 in lmbdas1:
    for lmbda2 in lmbdas2:
        paramList.append([lmbda1, lmbda2])

logging.info("Using "  + str(sampleSize) + " examples for model selection")
logging.info("List of lambda1s " + str(lmbdas1))
logging.info("List of lambda2s " + str(lmbdas2))

for noise in noises:
    outputDir = PathDefaults.getOutputDir()
    graphFilename = outputDir + "influence/SyntheticExamples_n=" + str(noise) + ".spg"
    graph = SparseGraph.load(graphFilename)

    #WEIRD ERROR ALERT: Normalisation makes errors much worse 
    #preprocessor = Standardiser()
    #vList = graph.getVertexList()
    #V = vList.getVertices(graph.getAllVertexIds())
    #V = preprocessor.standardiseArray(V)
    #vList.setVertices(V)

    #Split graph into training and test
    allEdges = graph.getAllEdges()
    inds = numpy.random.permutation(allEdges.shape[0])
    trainInds = inds[0:sampleSize]
    testInds = inds[sampleSize:]

    #testInds = testInds[0:sampleSize]

    trainEdges = allEdges[trainInds, :]
    testEdges = allEdges[testInds, :]

    trainGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
    trainGraph.addEdges(trainEdges, graph.getEdgeValues(trainEdges))

    testGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
    testGraph.addEdges(testEdges, graph.getEdgeValues(testEdges))

    meanErrors, stdErrors = edgeLearner.cvModelSelection(trainGraph, paramList, paramFuncs, folds, errorFunc)
    print(meanErrors)
    print((numpy.argmin(meanErrors)))
    bestErrorIndex = numpy.argmin(meanErrors)
    logging.info("Model selection returned params = " + str(paramList[bestErrorIndex]))

    paramsFile = outputDir + "influence/EgoParamsLinear_n=" + str(noise)
    edgeLearner.saveParams(paramList[bestErrorIndex], paramFuncs, paramsFile)

    logging.info("Using " + str(testInds.shape[0]) + " examples for Ego Network evaluation")

    (mean, std) = edgeLearner.cvError(testGraph, paramList[bestErrorIndex], paramFuncs, folds, errorFunc)

    resultsFile = outputDir + "influence/EgoResults_n=" + str(noise)
    numpy.savez(resultsFile, mean, std)
    logging.info("Mean error: " + str(mean))
    logging.info("Std error: " + str(std))
