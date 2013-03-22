"""
Compare our clustering method and that of Ning et al. on the HIV data.
"""
import os
import sys
import errno
import logging
import numpy
from apgl.graph import *
from exp.clusterexp.ClusterExpHelper import ClusterExpHelper
import argparse
from exp.clusterexp.HIVIterGenerator import HIVIterGenerator

if __debug__: 
    raise RuntimeError("Must run python with -O flag")

#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
dataArgs = argparse.Namespace()
dataArgs.monthStep = 1
dataArgs.minGraphSize = 500

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
# init (reading/writting command line arguments)
#=========================================================================
#=========================================================================

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
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
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

# print args #
logging.info("Running on HIV")
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
generator = HIVIterGenerator(dataArgs.minGraphSize, dataArgs.monthStep)
numGraphs = generator.getNumGraphs()
logging.info("Total graphs in sequence: " + str(numGraphs))

#=========================================================================
#=========================================================================
# run
#=========================================================================
#=========================================================================
logging.info("Creating the exp-runner")
clusterExpHelper = ClusterExpHelper(generator.getIterator, remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
clusterExpHelper.algoArgs.k1 = 25
clusterExpHelper.algoArgs.k2s = [25, 50, 100, 200]
clusterExpHelper.algoArgs.k3s = [100, 200, 500, 1000, 1500]
clusterExpHelper.algoArgs.k4s = [100, 200, 500, 1000]
#clusterExpHelper.algoArgs.k3s = [1500]
clusterExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(clusterExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

clusterExpHelper.runExperiment()
