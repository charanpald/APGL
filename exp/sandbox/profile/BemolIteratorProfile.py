import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.IterativeSpectralClustering import * 
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.BemolData import getBemolGraphIterator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



class IterativeSpectralClusteringProfile(object):
    def __init__(self):
        pass

    def profileClusterFromIterator(self):
        dataDir = PathDefaults.getDataDir() + "cluster/"
        iterator = getBemolGraphIterator(dataDir)

        def toRun():
            it = 0
            for i in iterator:
                print it
                it += 1
        ProfileUtils.profile('toRun()', globals(), locals())

profiler = IterativeSpectralClusteringProfile()
profiler.profileClusterFromIterator() #19.7 


# python -c "execfile('exp/sandbox/profile/BemolIteratorProfile.py')"
