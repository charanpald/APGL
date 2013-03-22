"""
Compare our clustering method and that of Ning et al. on the Bemol data 
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
from exp.clusterexp.BemolData import BemolData
from exp.sandbox.GraphIterators import MaxComponentsIterator
import argparse

if __debug__: 
    raise RuntimeError("Must run python with -O flag")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
dataArgs = argparse.Namespace()
dataArgs.nbUser = 10000 # set to 'None' to have all users
dataArgs.nbPurchasesPerIt = 500 # set to 'None' to take all the purchases
                                      # per date
dataArgs.startingIteration = 500    
dataArgs.endingIteration = 600 # set to 'None' to have all iterations
dataArgs.stepSize = 1 #This is the step in the number of weeks 
dataArgs.maxComponents = None

# Arguments related to the algorithm
# If one arg is not set, default from ClusterExpHelper.py is used
defaultAlgoArgs = argparse.Namespace()
#defaultAlgoArgs.runIASC = True
#defaultAlgoArgs.runExact = True
#defaultAlgoArgs.runModularity = True
#defaultAlgoArgs.runNystrom = True
#defaultAlgoArgs.runNing = True

defaultAlgoArgs.k1 = 100
defaultAlgoArgs.k2s = [100, 200, 500]
defaultAlgoArgs.k3s = [500, 1000, 2000, 5000]
defaultAlgoArgs.k4s = [200, 500, 1000, 2000]

defaultAlgoArgs.T = 20 

#=========================================================================
#=========================================================================
# usefull
#=========================================================================
#=========================================================================
# val should be a string at the beginning
def isIntOrNone(string):
    if string == "None":
        return None
    elif string.lstrip("+").isdigit():
        return int(string)
    else:
        msg = string + " is not a positive integer"
        raise argparse.ArgumentTypeError(msg)
    
#=========================================================================
#=========================================================================
# init (reading/writting command line arguments)
#=========================================================================
#=========================================================================

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
dataParser.add_argument("--nbUser", type=isIntOrNone, help="The graph is based on the $nbUser$ first users (default: %(default)s)", default=dataArgs.nbUser)
dataParser.add_argument("--nbPurchasesPerIt", type=isIntOrNone, help="Maximum number of purchases per iteration (default: %(default)s)", default=dataArgs.nbPurchasesPerIt)
dataParser.add_argument("--startingIteration", type=int, help="At which iteration to start clustering algorithms (default: %(default)s)", default=dataArgs.startingIteration)
dataParser.add_argument("--endingIteration", type=isIntOrNone, help="At which iteration to end clustering algorithms (default: %(default)s)", default=dataArgs.endingIteration)
dataParser.add_argument("--stepSize", type=int, help="Number of iterations between each clustering (default: %(default)s)", default=dataArgs.stepSize)
dataParser.add_argument("--maxComponents", type=isIntOrNone, help="Graphs with more than MAXCOMPONENTS components are filtered out (default: %(default)s)", default=dataArgs.maxComponents)
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, ClusterExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

#dataArgs.extendedDirName = "Bemol/"
dataArgs.extendedDirName = "Bemol_nbU=" + str(dataArgs.nbUser) + "_nbPurchPerIt=" + str(dataArgs.nbPurchasesPerIt) + "_startIt=" + str(dataArgs.startingIteration) + "_endIt=" + str(dataArgs.endingIteration) + "_maxComponents=" + str(dataArgs.maxComponents) + "/"

# seed #
numpy.random.seed(21)

# printing options #
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
dataDir = PathDefaults.getDataDir() + "cluster/"

def getIterator():
    bemolIterator = BemolData.getGraphIterator(dataDir, dataArgs.nbUser, dataArgs.nbPurchasesPerIt)
    if dataArgs.maxComponents:
        bemolIterator = MaxComponentsIterator(bemolIterator, dataArgs.maxComponents)
    return itertools.islice(bemolIterator, dataArgs.startingIteration, dataArgs.endingIteration, dataArgs.stepSize)

#logging.info("Computing the number of iterations")
#numGraphs = 0
#for W in getIterator():
#    numGraphs += 1
#    logging.info(str(numGraphs) + "\t" + str(W.shape))
#    if __debug__:
#        Parameter.checkSymmetric(W)
#logging.info("Number of iterations: " + str(numGraphs))


#=========================================================================
#=========================================================================
# run
#=========================================================================
#=========================================================================

logging.info("Creating the exp-runner")
clusterExpHelper = ClusterExpHelper(getIterator, remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
clusterExpHelper.printAlgoArgs()

#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(clusterExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

clusterExpHelper.runExperiment()

# to run
# python -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python2.7 -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python3 -c "exec(open('exp/clusterexp/BemolExperiment.py').read())"
# python3 exp/clusterexp/BemolExperiment.py --help

