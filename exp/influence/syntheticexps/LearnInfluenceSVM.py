"""
Here, we will create a number of graphs created under the same souce and classify.
This is part of the synthetic experiments, where we predict decays using the SVM. 
"""


from apgl.graph import *
from apgl.util import *
from apgl.data import *
from apgl.predictors import *
from apgl.influence.GreedyInfluence import GreedyInfluence
import numpy
import logging
import random
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
random.seed(22)
numpy.random.seed(22)

numVertices = 500
numFeatures = 5
numGraphs = 50

#ks = range(10, 100, 10)
ks = list(range(10, 310, 10))
maxK = max(ks)
noises = [0.05, 0.1, 0.15, 0.2]

outputDir = PathDefaults.getOutputDir() + "influence/"

svm = LibSVM()
svm.setTermination(0.01)
svm.setSvmType("Epsilon_SVR")
edgeLearner = FlatEdgeLabelPredictor(svm)

influence = GreedyInfluence()
p = 0.05

cFilename = outputDir + "C.npy"
C = numpy.load(cFilename)
logging.info("Loaded matrix of coefficients from " + cFilename)

classifier = "Svm"

for noise in noises: 
    graphFilename = outputDir + "SyntheticExamples_n=" + str(noise) + ".spg"
    graph = SparseGraph.load(graphFilename)
    
    preprocessor = Standardiser()
    vList = graph.getVertexList()
    V = vList.getVertices(graph.getAllVertexIds())
    V = preprocessor.standardiseArray(V)
    vList.setVertices(V)

    paramsFile = outputDir + classifier + "ParamsLinear_n=" + str(noise)
    paramsList = edgeLearner.loadParams(paramsFile)

    print(paramsList)

    for params in paramsList:
        paramFunction = getattr(svm, params[1])
        paramFunction(params[2])

    logging.info(svm)
    edgeLearner.learnModel(graph)

    influenceErrors = numpy.zeros((numGraphs, len(ks)))
    predictionErrors = numpy.zeros(numGraphs) 

    for i in range(numGraphs):
        Util.printIteration(i, 1, numGraphs)
        verticies = numpy.random.rand(numVertices, numFeatures)
        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(verticies)

        #Graph has noiseless edges, graph2 has noise 
        graph = SparseGraph(vList, False)
        graph2 = SparseGraph(vList, False) 

        generator = ErdosRenyiGenerator(graph)
        graph = generator.generateGraph(p)

        allEdges = graph.getAllEdges()
        logging.info("Number of edges: " + str(allEdges.shape[0]))

        egos = vList.getVertices(allEdges[:, 0])
        alters = vList.getVertices(allEdges[:, 1])
        alterCs = numpy.dot(egos, C)
        yReal = numpy.sum(alters * alterCs, 1)

        y = yReal + numpy.random.randn(allEdges.shape[0])*noise
        y[y>1] = 1
        y[y<0] = 0.01

        #For graph2 make prediction using SVM
        vList = graph.getVertexList()
        V = vList.getVertices(graph.getAllVertexIds())
        V = preprocessor.standardiseArray(V)
        vList.setVertices(V)
        
        predY = edgeLearner.predictEdges(graph2, allEdges)
        predY[predY>1] = 1
        predY[predY<0] = 0.01

        predictionErrors[i] = Evaluator.rootMeanSqError(y, predY)

        #Now assign the percolation decays to the edges for both graphs
        graph.addEdges(allEdges, y)
        graph2.addEdges(allEdges, predY)

        logging.info("min(y)= " + str(numpy.min(y)))
        logging.info("max(y)= " + str(numpy.max(y)))

        P = graph.maxProductPaths()
        P = P + numpy.eye(numVertices)

        P2 = graph2.maxProductPaths()
        P2 = P2 + numpy.eye(numVertices)

        logging.info("Computing max influence for learnt percolations")
        inds = numpy.array(influence.maxInfluence(P, maxK))

        logging.info("Computing max influence for real percolations")
        inds2 = numpy.array(influence.maxInfluence(P2, maxK))

        for j in range(len(ks)):
            k = ks[j]
            #print(numpy.sum(numpy.max(P[inds[0:k], :], 0)))
            influenceErrors[i, j] = numpy.setdiff1d(inds[0:k], inds2[0:k]).shape[0]/float(k)

    meanInfluenceErrors = numpy.mean(influenceErrors, 0)
    stdInfluenceErrors = numpy.std(influenceErrors, 0)
    logging.info(meanInfluenceErrors)
    logging.info(stdInfluenceErrors)

    logging.info("Mean prediction error: " + str(numpy.mean(predictionErrors)))

    fileName = outputDir + "influenceErrors" + classifier + "_n=" + str(noise)
    numpy.savez(fileName, meanInfluenceErrors, stdInfluenceErrors, predictionErrors)
    logging.info("Saved influenceErrors to " + fileName)

#Load using a = numpy.load(fileName), then meanErrors = a["arr_0"] 