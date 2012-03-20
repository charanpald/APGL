import logging
import sys
import numpy
import scipy.stats

from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVRates import HIVRates

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(24)

class HIVEpidemicModelProfile():
    def __init__(self):
        #Total number of people in population
        M = 10000
        numInitialInfected = 50

        #The graph is one in which edges represent a contact
        undirected = True
        self.graph = HIVGraph(M, undirected)

        for i in range(M):
            if i < numInitialInfected:
                self.graph.getVertexList().setInfected(i, 0.0)


    def profileSimulate(self):
        #End time
        T = 300.0

        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, self.graph.getNumVertices())

        rates = HIVRates(self.graph, hiddenDegSeq)
        model = HIVEpidemicModel(self.graph, rates)
        model.setT(T)

        ProfileUtils.profile('model.simulate()', globals(), locals())

profiler = HIVEpidemicModelProfile()
profiler.profileSimulate()

#97.8
#81.3
#48.45