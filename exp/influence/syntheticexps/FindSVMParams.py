"""
Find SVM parameters using cross validation and then evaluate the error, and
save the results. 
"""
from apgl.data import *
from apgl.util import *
from apgl.predictors import *
from apgl.predictors.edge import *
from apgl.graph import * 
import logging
import sys
import numpy 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

noises = [0.0, 0.05, 0.1, 0.15, 0.2]
sampleSize = 10000
#sampleSize = 1000
folds = 5
kernel = "linear"
Cs = [0.03125, 0.0625, 0.125, 0.25, 0.5]
epsilons = [0.03125, 0.0625, 0.125, 0.25, 0.5]
kernelParams = [1]
errorCosts = [1.0]
errorFunc = Evaluator.rootMeanSqError

svm = LibSVM()
svm.setSvmType("Epsilon_SVR")

edgeLearner = FlatEdgeLabelPredictor(svm)

paramList = []
paramFuncs = [svm.setC, svm.setEpsilon]

for C in Cs:
    for epsilon in epsilons:
        paramList.append([C, epsilon])

logging.info("Using "  + str(sampleSize) + " examples for model selection")
logging.info("List of Cs " + str(Cs))
logging.info("List of kernels " + str(kernel))

for noise in noises:
    outputDir = PathDefaults.getOutputDir()
    graphFilename = outputDir + "influence/SyntheticExamples_n=" + str(noise) + ".spg"
    graph = SparseGraph.load(graphFilename)
    
    preprocessor = Standardiser()
    vList = graph.getVertexList()
    V = vList.getVertices(graph.getAllVertexIds())
    V = preprocessor.standardiseArray(V)
    vList.setVertices(V)

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

    paramsFile = outputDir + "influence/SvmParamsLinear_n=" + str(noise)
    edgeLearner.saveParams(paramList[bestErrorIndex], paramFuncs, paramsFile)

    logging.info("Using " + str(testInds.shape[0]) + " examples for SVM evaluation")

    (mean, std) = edgeLearner.cvError(testGraph, paramList[bestErrorIndex], paramFuncs, folds, errorFunc)

    resultsFile = outputDir + "influence/SvmResults_n=" + str(noise)
    numpy.savez(resultsFile, mean, std)
    logging.info("Mean error: " + str(mean))
    logging.info("Std error: " + str(std))
