"""
Write Pajek files of the transmissions graphs using the simulation. 

"""

import logging
import random
import sys
import time

from apgl.egograph.EgoSimulator import EgoSimulator
from apgl.egograph.EgoUtils import EgoUtils
from apgl.egograph.InfoExperiment import InfoExperiment
from apgl.egograph.SvmEgoSimulator import SvmEgoSimulator
from apgl.io import *
from apgl.util import *
import numpy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

examplesFileName = SvmInfoExperiment.getExamplesFileName()
egoFileName = "../../data/EgoData.csv"
alterFileName = "../../data/AlterData.csv"
numVertices =  SvmInfoExperiment.getNumVertices()
numVertices = 10000

infoProb = 0.1

graphTypes = ["SmallWorld", "ErdosRenyi"]
ps = [0.1, 0.003]
ks = [15, 15]

svmParamsFile = SvmInfoExperiment.getSvmParamsFileName()
sampleSize = SvmInfoExperiment.getNumSimulationExamples()

simulator = SvmEgoSimulator(examplesFileName)
CVal, kernel, kernelParamVal, errorCost = SvmInfoExperiment.loadSvmParams(svmParamsFile)
classifier = simulator.trainClassifier(CVal, kernel, kernelParamVal, errorCost, sampleSize)
preprocessor = simulator.getPreprocessor()

maxIterations = 5

eCsvReader = EgoCsvReader()
egoQuestionIds = eCsvReader.getEgoQuestionIds()

pajekWriter = PajekWriter()
simpleGraphWriter = SimpleGraphWriter()
vertexWriter = CsvVertexWriter()

for i in range(len(graphTypes)):
    graphType = graphTypes[i]
    p = ps[i]
    k = ks[i]

    outputDirectory = PathDefaults.getOutputDir()
    baseFileName = outputDirectory + "InfoGraph" + graphType
    graph = simulator.generateRandomGraph(egoFileName, alterFileName, numVertices, infoProb, graphType, p, k)

    #Notice that the data is preprocessed in the same way as the survey data
    egoSimulator = EgoSimulator(graph, classifier, preprocessor)

    totalInfo = numpy.zeros(maxIterations+1)
    totalInfo[0] = EgoUtils.getTotalInformation(graph)
    logging.info("Total number of people with information: " + str(totalInfo[0]))
    logging.info("--- Simulation Started ---")

    for i in range(0, maxIterations):
        logging.info("--- Iteration " + str(i) + " ---")
        graph = egoSimulator.advanceGraph()
        totalInfo[i+1] = EgoUtils.getTotalInformation(graph)
        logging.info("Total number of people with information: " + str(totalInfo[i+1]))

    transmissionGraph  = egoSimulator.getTransmissionGraph()
    pajekWriter.writeToFile(baseFileName, transmissionGraph)
    vertexWriter.writeToFile(baseFileName, transmissionGraph)
    simpleGraphWriter.writeToFile(baseFileName, transmissionGraph)
    logging.info("--- Simulation Finished ---")




