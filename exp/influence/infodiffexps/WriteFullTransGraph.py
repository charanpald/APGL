"""
Write out a complete transmission graph for the information diffusion data. 
"""

import svm
import logging
import random
import sys
import time
import numpy

from apgl.egograph import * 
from apgl.io import *
from apgl.util import *
from apgl.graph import *
from apgl.kernel import *
from apgl.data.Standardiser import Standardiser 
from apgl.predictors import *
from apgl.generator import * 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
random.seed(21)
numpy.random.seed(21)
startTime = time.time()

#Load up the decay graph 
graphFileName = InfoExperiment.getGraphFileName()
decayGraph  = SparseGraph.load(graphFileName)

defaultTol = 10**-5
allEdges = decayGraph.getAllEdges()
edgeDecays = decayGraph.getEdgeValues(allEdges)
#-1 is no transmission, +1 is transmission 
binaryEdges = numpy.array(edgeDecays > defaultTol, numpy.int)*2 -1

logging.info("Total number of transmisions: " + str(numpy.sum(binaryEdges[binaryEdges==1])))

#Center the decays values
logging.warn("Centering the edge decays for transmissions.")
meanDecay = numpy.mean(edgeDecays[binaryEdges==1]) 
logging.warn("Mean edge value :" + str(meanDecay))
edgeDecays[binaryEdges==1] = edgeDecays[binaryEdges==1] - numpy.mean(edgeDecays[binaryEdges==1])

logging.warn("About to modify (standardise) the vertices of the graph.")
preprocessor = Standardiser()
V = decayGraph.getVertexList().getVertices(decayGraph.getAllVertexIds())
V = preprocessor.standardiseArray(V)
decayGraph.getVertexList().setVertices(V)

#Take a subgraph
#edgeSampleSize = 20000
edgeSampleSize = 1000
transEdges = allEdges[0:edgeSampleSize, :]
transEdgeLabels = binaryEdges[0:edgeSampleSize]

transGraph = SparseGraph(decayGraph.getVertexList(), False)
transGraph.addEdges(transEdges, transEdgeLabels)
logging.info("Created graph of binary transmissions")
logging.info("Transmission graph: " + str(transGraph))

#A graph with transmission edges only and the corresponding decays 
decayGraph2 = SparseGraph(decayGraph.getVertexList(), False)
decayGraph2.addEdges(allEdges[binaryEdges==1], edgeDecays[binaryEdges==1])
decayGraph2EdgeValues = decayGraph2.getEdgeValues(decayGraph2.getAllEdges())
logging.info("Created transmissions graph with decays")
logging.info("Min decay " + str(numpy.min(decayGraph2EdgeValues)) + " max decay " + str(numpy.max(decayGraph2EdgeValues)))
logging.info("DecayGraph2: " + str(decayGraph2))

#Predict the vertices using the SVM model we used in the chatpter, then use
#the ego-network prediction
outputDir = PathDefaults.getOutputDir() + "diffusion/"
svmParamsFileName = SvmInfoExperiment.getSvmParamsFileName()
svm = LibSVM()
svm.loadParams(svmParamsFileName)
logging.info("Loaded SVM params : " + str(svm))
edgeLearner = FlatEdgeLabelPredictor(svm)
edgeLearner.learnModel(transGraph)

#Load params for the decay predictor
lmbda1 = 0.001
lmbda2 = 0.001
kernel = LinearKernel()
alterRegressor = PrimalRidgeRegression(lmbda1)
egoRegressor = KernelShiftRegression(kernel, lmbda2)
edgeLearner2 = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

paramsFile = outputDir + "EgoNetInfoParams.pkl"
paramsList = edgeLearner2.loadParams(paramsFile)

#Translate saved parameters into a paramList and paramFuncs
print(paramsList)
params = []
paramFuncs = []

for p in paramsList:
    if p[0] == "apgl.predictors.PrimalRidgeRegression":
        paramFunction = getattr(alterRegressor, p[1])
        paramFuncs.append(paramFunction)
        params.append(p[2])

    if p[0] == "apgl.predictors.KernelShiftRegression":
        paramFunction = getattr(egoRegressor, p[1])
        paramFuncs.append(paramFunction)
        params.append(p[2])

    if p[0] == "apgl.kernel.GaussianKernel":
        kernel = GaussianKernel()
        egoRegressor.setKernel(kernel)
        paramFunction = getattr(kernel, p[1])
        paramFuncs.append(paramFunction)
        params.append(p[2])

for j in range(len(params)):
    paramFuncs[j](params[j])

#Just to test the decay prediction 
edgeLearner2.learnModel(decayGraph2)

#Predict edges here:
predDecays = edgeLearner2.predictEdges(decayGraph2, decayGraph2.getAllEdges())
print((Evaluator.rootMeanSqError(decayGraph2.getEdgeValues(decayGraph2.getAllEdges()), predDecays)))
print((predDecays[0:20]))
print((decayGraph2.getEdgeValues(decayGraph2.getAllEdges())[0:20]))
#What is going on - why are results so bad? 

#Now, let's generate simulated graphs and learn the percolations 
#Size of the simulated graph
numVertices = 1000
#numVertices = 200
ps = [0.2, 0.2, float(30)/numVertices]
ks = [15, 50]

generators = [SmallWorldGenerator(ps[0], ks[0]), SmallWorldGenerator(ps[1], ks[1])]
generators.append([ErdosRenyiGenerator(ps[2])])

dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
egoFileName = dataDir + "EgoData.csv"
alterFileName = dataDir + "AlterData.csv"
outputDirectory = PathDefaults.getOutputDir()

simulator = EgoNetworkSimulator(transGraph, edgeLearner)
infoProb = 0.0

for i in range(len(generators)):
    vList = VertexList(numVertices, 0)
    graph = SparseGraph(vList)
    graph = generators[i].generate(graph)
        
    baseFileName = outputDirectory + "influence/FullTransGraph" + str(generators[i]) + ".spg"
    randomGraph = simulator.generateRandomGraph(egoFileName, alterFileName, infoProb, graph)

    #Notice that the data is preprocessed in the same way as the survey data
    vList = randomGraph.getVertexList()
    V = vList.getVertices(randomGraph.getAllVertexIds())
    V = V[:, 0:V.shape[1]-1]
    V = preprocessor.standardiseArray(V)
    vList.replaceVertices(V)

    edges = randomGraph.getAllEdges()
    inverseEdges = numpy.c_[edges[:, 1], edges[:, 0]]
    allEdges = numpy.r_[edges, inverseEdges]

    y = edgeLearner.predictEdges(randomGraph, allEdges)

    transmissionGraph = SparseGraph(vList, False)
    transmissionGraph.addEdges(allEdges[y==1, :])
    logging.info("Output transmission graph with " + str(transmissionGraph.getNumEdges()) + " edges")

    #Now we use edgeLearner2 to predict decays
    logging.info("Making predictions of edge decays")
    predDecays = edgeLearner2.predictEdges(transmissionGraph, allEdges[y==1, :]) + meanDecay
    transmissionGraph.addEdges(allEdges[y==1, :], predDecays)
    transmissionGraph.save(baseFileName)

    logging.info("Number of vertices " + str(transmissionGraph.getNumVertices()) + " num edges " + str(transmissionGraph.getNumEdges()))
    allEdges = transmissionGraph.getAllEdges()
    edgeValues = transmissionGraph.getEdgeValues(allEdges)
    logging.info("Min decay " + str(numpy.min(edgeValues)) + " max decay " + str(numpy.max(edgeValues)))

