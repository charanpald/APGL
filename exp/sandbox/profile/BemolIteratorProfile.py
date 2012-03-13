import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from apgl.sandbox.IterativeSpectralClustering import * 
from apgl.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from apgl.util.PathDefaults import PathDefaults
from apgl.clusterexp.BemolData import getBemolGraphIterator

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


# python -c "execfile('apgl/sandbox/profile/BemolIteratorProfile.py')"
