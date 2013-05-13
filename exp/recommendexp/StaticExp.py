

"""
Run some experiments on a static dataset.
"""

import os
import sys
import errno
import logging
import numpy
import argparse
from apgl.graph import *
from exp.recommendexp.RecommendExpHelper import RecommendExpHelper
from exp.recommendexp.StaticDataset import StaticMatrixMarketDataset

if __debug__:
    raise RuntimeError("Must run python with -O flag")

numpy.random.seed(21)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=60)
numpy.seterr("raise", under="ignore")

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()

# init (reading/writting command line arguments)
# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
dataParser.add_argument("--train", dest="train_file", help="training datafile (MatrixMarket format)")
dataParser.add_argument("--test", dest="test_file", help="test datafile (MatrixMarket format)")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RecommendExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "StaticDataset"

# print args #
logging.info("Running on {0} / {1}".format(dataArgs.train_file,dataArgs.test_file))
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

# data
generator = StaticMatrixMarketDataset(dataArgs.train_file,dataArgs.test_file)

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
