"""
Run the clustering experiments on the High Energy Physics citation network
provided at http://snap.stanford.edu/data/cit-HepTh.html
"""

import os
import sys
import errno
import itertools
import logging
import numpy
import argparse
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
from exp.clusterexp.CitationIterGenerator import CitationIterGenerator 

if __debug__: 
    raise RuntimeError("Must run python with -O flag")

#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
dataArgs = argparse.Namespace()
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

#=========================================================================
#=========================================================================
# useful
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
dataParser.add_argument("--startingIteration", type=int, help="At which iteration to start clustering algorithms", default=dataArgs.startingIteration)
dataParser.add_argument("--endingIteration", type=isIntOrNone, help="At which iteration to end clustering algorithms", default=dataArgs.endingIteration)
dataParser.add_argument("--stepSize", type=int, help="Number of iterations between each clustering", default=dataArgs.stepSize)
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, ClusterExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "Citation"

# seed #
numpy.random.seed(21)

# printing options #
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(levelname)s (%(asctime)s):%(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

# print args #
logging.info("Running on Citation")
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
generator = CitationIterGenerator()

def getIterator():
    return itertools.islice(generator.getIterator(), dataArgs.startingIteration, dataArgs.endingIteration, dataArgs.stepSize)


#=========================================================================
#=========================================================================
# run
#=========================================================================
#=========================================================================
logging.info("Creating the exp-runner")
clusterExpHelper = ClusterExpHelper(getIterator, remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
clusterExpHelper.algoArgs.T = 20 
clusterExpHelper.algoArgs.k1 = 50
clusterExpHelper.algoArgs.k2s = [50, 100, 200, 500]
clusterExpHelper.algoArgs.k3s = [1000, 2000, 5000]
clusterExpHelper.printAlgoArgs()

#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(clusterExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

clusterExpHelper.runExperiment()
