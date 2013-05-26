import os 
import numpy
import logging
import sys
import argparse
import time
import errno
from datetime import datetime
from apgl.util.ProfileUtils import ProfileUtils
from exp.recommendexp.MovieLensDataset import MovieLensDataset
from exp.recommendexp.RecommendExpHelper import RecommendExpHelper

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MovieLensExpProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        
    def profileRunExperiment(self):
        
        def run(): 
            dataArgs = argparse.Namespace()
            dataArgs.maxIter = 3 
            #Set iterStartDate to None for all iterations 
            #dataArgs.iterStartTimeStamp = None 
            dataArgs.iterStartTimeStamp = time.mktime(datetime(2005,1,1).timetuple())
            generator = MovieLensDataset(maxIter=dataArgs.maxIter, iterStartTimeStamp=dataArgs.iterStartTimeStamp)        
            
            defaultAlgoArgs = argparse.Namespace()
            defaultAlgoArgs.ks = numpy.array(2**numpy.arange(6, 7, 0.5), numpy.int)
            defaultAlgoArgs.svdAlg = "rsvd"   
            defaultAlgoArgs.runSoftImpute = True
            
            dataParser = argparse.ArgumentParser(description="", add_help=False)
            dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
            devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
            
            dataArgs.extendedDirName = ""
            dataArgs.extendedDirName += "MovieLensDataset"
            
            recommendExpHelper = RecommendExpHelper(generator.getTrainIteratorFunc, generator.getTestIteratorFunc, remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
            recommendExpHelper.printAlgoArgs()
            #    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
            try:
                os.makedirs(recommendExpHelper.resultsDir)
            except OSError as err:
                if err.errno != errno.EEXIST:
                    raise
            
            recommendExpHelper.runExperiment()
            
        ProfileUtils.profile('run()', globals(), locals())    

profiler = MovieLensExpProfile()
profiler.profileRunExperiment()
