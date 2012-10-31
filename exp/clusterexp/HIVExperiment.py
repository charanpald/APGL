"""
Compare our clustering method and that of Ning et al. on the HIV data.
"""

import os
import sys
import errno
import itertools
import logging
import numpy
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from apgl.util.Parameter import Parameter
import argparse
from exp.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.DateUtils import DateUtils
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator


#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
dataArgs = argparse.Namespace()
dataArgs.startYear = 1900
dataArgs.daysInMonth = 30
dataArgs.monthStep = 3
dataArgs.minGraphSize = 500

# Arguments related to the algorithm
# If one arg is not set, default from ClusterExpHelper.py is used
defaultAlgoArgs = argparse.Namespace()
#defaultAlgoArgs.runIASC = True
#defaultAlgoArgs.runExact = True
#defaultAlgoArgs.runModularity = True
#defaultAlgoArgs.runNystrom = True
#defaultAlgoArgs.runNing = True

#defaultAlgoArgs.k1 = 5
#defaultAlgoArgs.k2 = 10
#defaultAlgoArgs.k3 = 15
    
#=========================================================================
#=========================================================================
# init (reading/writting command line arguments)
#=========================================================================
#=========================================================================

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
dataParser.add_argument("--startYear", type=int, help="", default=dataArgs.startYear)
dataParser.add_argument("--daysInMonth", type=int, help="", default=dataArgs.daysInMonth)
dataParser.add_argument("--monthStep", type=int, help="", default=dataArgs.monthStep)
dataParser.add_argument("--minGraphSize", type=int, help="", default=dataArgs.minGraphSize)
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, ClusterExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "HIV"


# seed #
numpy.random.seed(21)

# printing options #
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(levelname)s (%(asctime)s):%(message)s')
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")



# print args #
logging.info("Running on Bemol")
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))




#=========================================================================
#=========================================================================
# data
#=========================================================================
#=========================================================================
#Start off with the HIV data 
hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph()
fInds = hivReader.getIndicatorFeatureIndices()

#The set of edges indexed by zeros is the contact graph
#The ones indexed by 1 is the infection graph
edgeTypeIndex1 = 0
edgeTypeIndex2 = 1
sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)
sGraphContact = sGraphContact.union(sGraphInfect)
graph = sGraphContact

#Find max component
components = graph.findConnectedComponents()
graph = graph.subgraph(list(components[0]))
logging.debug(graph)

detectionIndex = fInds["detectDate"]
vertexArray = graph.getVertexList().getVertices()
detections = vertexArray[:, detectionIndex]

dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), dataArgs.daysInMonth*dataArgs.monthStep))
dayList.append(numpy.max(detections))

subgraphIndicesList = []

#Generate subgraph indices list 
for i in dayList:
    logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, dataArgs.startYear)))
    subgraphIndices = numpy.nonzero(detections <= i)[0]
    if subgraphIndices.shape[0] >= dataArgs.minGraphSize: 
        subgraphIndicesList.append(subgraphIndices)

#subgraphIndicesList2 = []
#for i in range(5):
#    subgraphIndicesList2.append(subgraphIndicesList[i])
#subgraphIndicesList = subgraphIndicesList2

def getIterator():
    return IncreasingSubgraphListIterator(graph, subgraphIndicesList)
    
numGraphs = len(subgraphIndicesList)
logging.info("number of iterations: " + str(numGraphs))

#=========================================================================
#=========================================================================
# run
#=========================================================================
#=========================================================================
logging.info("Creating the exp-runner")
clusterExpHelper = ClusterExpHelper(getIterator, numGraphs, remainingArgs, defaultAlgoArgs)
clusterExpHelper.printAlgoArgs()

clusterExpHelper.extendResultsDir(dataArgs.extendedDirName)
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(clusterExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

clusterExpHelper.runExperiment()
