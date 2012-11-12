import os
import sys
import errno
import itertools
import logging
import numpy
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.BemolData import BemolData
from apgl.util.ProfileUtils import ProfileUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class BemolDataProfile(object):
    def __init__(self):
        dataDir = PathDefaults.getDataDir() + "cluster/"
        nbUser = 2000 # set to 'None' to have all users
        nbPurchasesPerIt = 50 # set to 'None' to take all the purchases
                                              # per date
        startingIteration = 20
        endingIteration = None # set to 'None' to have all iterations
        stepSize = 10    
        
        iterator = itertools.islice(BemolData.getGraphIterator(dataDir, nbUser, nbPurchasesPerIt), startingIteration, endingIteration, stepSize)
        self.iterator = iterator 

    def profileIterator(self):

        def run(): 
            subgraphIndicesList = []
            for W in self.iterator: 
                subgraphIndicesList.append(range(W.shape[0])) 

        ProfileUtils.profile('run()', globals(), locals())

profiler = BemolDataProfile() #30s
profiler.profileIterator() 
