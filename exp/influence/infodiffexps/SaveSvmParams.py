import svm
import logging
import sys
import numpy
import random
from apgl.egograph import *
from apgl.graph import *
from apgl.predictors import *
from apgl.predictors.edge import *
from apgl.data import *
from apgl.util import *
from apgl.kernel import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)
random.seed(21)

#We want to learn on the graph of transmissions with decays
#Load up the decay graph
graphFileName = InfoExperiment.getGraphFileName()
decayGraph  = SparseGraph.load(graphFileName)

defaultTol = 10**-5
allEdges = decayGraph.getAllEdges()
edgeDecays = decayGraph.getEdgeValues(allEdges)
#-1 is no transmission, +1 is transmission
binaryEdges = numpy.array(edgeDecays > defaultTol, numpy.int)*2 -1

#Center the decays values
logging.warn("Centering the edge decays for transmissions.")
logging.warn("Mean edge value :" + str(numpy.mean(edgeDecays[binaryEdges==1])))
edgeDecays[binaryEdges==1] = edgeDecays[binaryEdges==1] - numpy.mean(edgeDecays[binaryEdges==1])

logging.info("Total number of transmisions: " + str(numpy.sum(binaryEdges[binaryEdges==1])))
logging.warn("About to modify (standardise) the vertices of the graph.")
preprocessor = Standardiser()
V = decayGraph.getVertexList().getVertices(decayGraph.getAllVertexIds())
V = preprocessor.standardiseArray(V)
decayGraph.getVertexList().setVertices(V)

#Take a subgraph
#edgeSampleSize = 1000
#allEdges = allEdges[0:edgeSampleSize, :]
#binaryEdges = binaryEdges[0:edgeSampleSize]

#A graph with transmission edges only and the corresponding decays
decayGraph2 = SparseGraph(decayGraph.getVertexList(), False)
decayGraph2.addEdges(allEdges[binaryEdges==1], edgeDecays[binaryEdges==1])
decayGraph2EdgeValues = decayGraph2.getEdgeValues(decayGraph2.getAllEdges())
logging.info("Created transmissions graph with decays " + str(decayGraph2))
logging.info("Min decay " + str(numpy.min(decayGraph2EdgeValues)) + " max decay " + str(numpy.max(decayGraph2EdgeValues)))

logging.info("Find all ego networks")
trees = decayGraph2.findTrees()
subgraphSize = 10000
subgraphIndices = []

for i in range(len(trees)):
    subgraphIndices.extend(trees[i])

    if len(subgraphIndices) > subgraphSize:
        logging.info("Chose " + str(i) + " ego networks.")
        break

decayGraph2 = decayGraph2.subgraph(subgraphIndices)
logging.info("Taking random subgraph of size " + str(decayGraph2.getNumVertices()))

folds = 3
sampleSize = decayGraph2.getNumEdges()

Cs = 2.0**numpy.arange(-4,1)
epsilons = 2.0**numpy.arange(-4, -2)
sigmas = 2.0**numpy.arange(-8,0)
sigmas = [("gaussian", x) for x in sigmas]

logging.info("Cs = " + str(Cs))
logging.info("epsilons = " + str(epsilons))
logging.info("sigmas = " + str(sigmas))

svm = LibSVM()
svm.setTermination(0.01)
svm.setSvmType("Epsilon_SVR")
predictor = FlatEdgeLabelPredictor(svm)

simulator = EgoNetworkSimulator(decayGraph2, predictor)
errorFunc = Evaluator.rootMeanSqError

paramList = []
paramFuncs = [svm.setC, svm.setEpsilon]

#First just use the linear kernel
for i in Cs:
    for j in epsilons:
        paramList.append([i, j])

params, paramFuncs, error = simulator.modelSelection(paramList, paramFuncs, folds, errorFunc, sampleSize)

#Now try the RBF kernel
svm.setKernel("gaussian", 0.1)

paramFuncs2 = [svm.setC, svm.setEpsilon, svm.setKernel]
paramList2 = []

for i in Cs:
    for j in epsilons:
        for k in sigmas:
            paramList2.append([i, j, k])

params2, paramFuncs2, error2 = simulator.modelSelection(paramList2, paramFuncs2, folds, errorFunc, sampleSize)

if error2 < error:
    params = params2
    paramFuncs = paramFuncs2

outputDir = PathDefaults.getOutputDir() + "diffusion/"
paramsFile = outputDir + "SvmInfoParams.pkl"
(means, vars) = simulator.evaluateClassifier(params, paramFuncs, folds, errorFunc, sampleSize)

logging.info("Evaluated classifier with mean errors " + str(means))
simulator.getClassifier().saveParams(params, paramFuncs, paramsFile)