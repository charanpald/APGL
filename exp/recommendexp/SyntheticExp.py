

"""
Run some experiments using SyntheticDataset1 
"""

import os
import sys
import errno
import logging
import numpy
import argparse
from apgl.graph import *
from exp.recommendexp.RecommendExpHelper import RecommendExpHelper
from exp.recommendexp.SyntheticDataset1 import SyntheticDataset1

#if __debug__: 
#    raise RuntimeError("Must run python with -O flag")

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=150)
numpy.seterr("raise", under="ignore")

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array(2**numpy.arange(3, 7, 0.5), numpy.int)
defaultAlgoArgs.rhos = numpy.linspace(0.5, 0.0, 10)  
defaultAlgoArgs.folds = 3

# init (reading/writting command line arguments)
# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RecommendExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "SyntheticDataset1"

# print args #
logging.info("Running on SyntheticDataset1")
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

# data
generator = SyntheticDataset1()

# run
logging.info("Creating the exp-runner")
recommendExpHelper = RecommendExpHelper(generator.getTrainIteratorFunc(), generator.getTestIteratorFunc(), remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
recommendExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(recommendExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

recommendExpHelper.runExperiment()
