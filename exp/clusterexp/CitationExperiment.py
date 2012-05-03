"""
Run the clustering experiments on the High Energy Physics citation network
provided at http://snap.stanford.edu/data/cit-HepTh.html
"""

import numpy 
import sys 
import logging
import itertools
from exp.clusterexp.CitationIterGenerator import CitationIterGenerator 
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)

generator = CitationIterGenerator()

def getIterator():
    #return itertools.islice(generator.getIterator(), 70, 115, 1)
    return generator.getIterator()

#Count the number of graphs
iter = getIterator()
numGraphs = 0
for W in iter:
    numGraphs += 1

logging.info("Total graphs in sequence: " + str(numGraphs))

datasetName = "Citation"

clusterExpHelper = ClusterExpHelper(getIterator, datasetName, numGraphs)
clusterExpHelper.runIASC = False
clusterExpHelper.runExact = False
clusterExpHelper.runModularity = True
clusterExpHelper.runNing = True
clusterExpHelper.k1 = 25
clusterExpHelper.k2 = 2*clusterExpHelper.k1

clusterExpHelper.runExperiment()
