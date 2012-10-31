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
from apgl.util.Parameter import Parameter
import argparse


#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
dataArgs = argparse.Namespace()
dataArgs.nbUser = 1000 # set to 'None' to have all users
dataArgs.nbPurchasesPerIt = 10 # set to 'None' to take all the purchases
                                      # per date
dataArgs.startingIteration = 0
dataArgs.endingIteration = None # set to 'None' to have all iterations
dataArgs.stepSize = 1


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
dataParser.add_argument("--nbUser", type=isIntOrNone, help="The graph is based on the $nbUser$ first users", default=dataArgs.nbUser)
dataParser.add_argument("--nbPurchasesPerIt", type=isIntOrNone, help="Maximum number of purchases per iteration", default=dataArgs.nbPurchasesPerIt)
dataParser.add_argument("--startingIteration", type=int, help="At which iteration to start clustering algorithms", default=dataArgs.startingIteration)
dataParser.add_argument("--endingIteration", type=isIntOrNone, help="At which iteration to end clustering algorithms", default=dataArgs.endingIteration)
dataParser.add_argument("--stepSize", type=int, help="Number of iterations between each clustering", default=dataArgs.stepSize)
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, ClusterExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "Bemol"
dataArgs.extendedDirName += "__nbU_" + str(dataArgs.nbUser)
dataArgs.extendedDirName += "__nbPurchPerIt_" + str(dataArgs.nbPurchasesPerIt)
dataArgs.extendedDirName += "__startIt_" + str(dataArgs.startingIteration)
dataArgs.extendedDirName += "__endIt_" + str(dataArgs.endingIteration)
dataArgs.extendedDirName += "/"


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
dataDir = PathDefaults.getDataDir() + "cluster/"
def getIterator():
    return itertools.islice(BemolData.getGraphIterator(dataDir, dataArgs.nbUser, dataArgs.nbPurchasesPerIt), dataArgs.startingIteration, dataArgs.endingIteration, dataArgs.stepSize)

logging.info("Computing the number of iteration")
numGraphs = 0
for W in getIterator():
    numGraphs += 1
#    logging.info(str(numGraphs) + "\t" + str(W.shape))
#    if __debug__:
#        Parameter.checkSymmetric(W)
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

# to run
# python -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python2.7 -c "execfile('exp/clusterexp/BemolExperiment.py')"
# python3 -c "exec(open('exp/clusterexp/BemolExperiment.py').read())"
# python3 exp/clusterexp/BemolExperiment.py --help

