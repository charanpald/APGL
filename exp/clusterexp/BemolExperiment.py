"""
Compare our clustering method and that of Ning et al. on the Bemol data 
"""
import itertools
import sys
import logging
import numpy
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.clusterexp.BemolData import BemolData
from apgl.util.Parameter import Parameter

numpy.random.seed(21)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(levelname)s (%(asctime)s):%(message)s')
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

resultsDir = PathDefaults.getOutputDir() + "cluster/"
dataDir = PathDefaults.getDataDir() + "cluster/"

nb_user = 1000
nb_purchases_per_it = -1
nb_purchases_per_it = 100

startingIteration = 0
endingIteration = None # set to None to have all iterations
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
clusterExpHelper.runNystrom = False
clusterExpHelper.runNing = True
clusterExpHelper.k1 = 25
clusterExpHelper.k2 = 8*clusterExpHelper.k1
clusterExpHelper.runExperiment()

# to run
# python -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python2.7 -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python3 -c "exec(open('exp/clusterexp/BemolExperiment.py').read())"

