"""
Compare our clustering method and that of Ning et al. on the Bemol data 
"""
import itertools
import sys
import logging
import numpy
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from apgl.clusterexp.ClusterExpHelper import ClusterExpHelper
from apgl.clusterexp.BemolData import BemolData
from apgl.util.Parameter import Parameter

numpy.random.seed(21)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(levelname)s (%(asctime)s):%(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise")

resultsDir = PathDefaults.getOutputDir() + "cluster/"
dataDir = PathDefaults.getDataDir() + "cluster/"

nb_user = 4000
nb_purchases_per_it = -1
nb_purchases_per_it = 100

startingIteration = 0
endingIteration = 100 # put a big number to have all iterations
stepSize = 1

def getIterator():
    return itertools.islice(BemolData.getGraphIterator(dataDir, nb_user, nb_purchases_per_it), startingIteration, endingIteration, stepSize)

numGraphs = 0
for W in getIterator():
    numGraphs += 1
#    print numGraphs
#    if __debug__:
#        Parameter.checkSymmetric(W)


datasetName = "Bemol"

clusterExpHelper = ClusterExpHelper(getIterator, datasetName, numGraphs)
clusterExpHelper.runIASC = False
clusterExpHelper.runExact = False
clusterExpHelper.runModularity = False
clusterExpHelper.runNystrom = True
clusterExpHelper.runNing = False
clusterExpHelper.k1 = 25
clusterExpHelper.k2 = 8*clusterExpHelper.k1
clusterExpHelper.k1 = 100
clusterExpHelper.runExperiment()

# to run
# python -c "execfile('apgl/clusterexp/BemolExperiment.py')"
# python2.7 -c "execfile('apgl/clusterexp/BemolExperiment.py')"


