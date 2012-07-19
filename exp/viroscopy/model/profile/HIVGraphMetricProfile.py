#Try to improve the speed of graph metrics 

import logging
import sys
import numpy
import scipy
import scipy.stats 
from exp.sandbox.GraphMatch import GraphMatch
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics2
from apgl.graph import *
from apgl.util import *
from apgl.generator import ErdosRenyiGenerator 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

class HIVGraphMetricsProfile():
    def __init__(self):
        #Total number of people in population
        self.M = 1000
        numInitialInfected = 5

        #The graph is one in which edges represent a contact
        undirected = True
        self.graph = HIVGraph(self.M, undirected)

        for i in range(self.M):
            vertex = self.graph.getVertex(i)

            #Set the infection time of a number of individuals to 0
            if i < numInitialInfected:
                vertex[HIVVertices.stateIndex] = HIVVertices.infected
            

        p = 0.01
        generator = ErdosRenyiGenerator(p)
        self.graph = generator.generate(self.graph)
        
        perm1 = numpy.random.permutation(self.M)
        perm2 = numpy.random.permutation(self.M)        
        
        sizes = [200, 300, 500, 1000]
        self.summary1 = [] 
        self.summary2 = [] 

        for size in sizes: 
            self.summary1.append(self.graph.subgraph(perm1[0:size]))
            self.summary2.append(self.graph.subgraph(perm2[0:int(size/10)]))
        
        print(self.graph)

    def profileDistance(self): 
        times = numpy.arange(len(self.summary1))
        #metrics = HIVGraphMetrics2(times, GraphMatch("RANK"))
        metrics = HIVGraphMetrics2(times, GraphMatch("U"))
        
        #Can try RANK and Umeyama algorithm - Umeyama is faster      
        self.summary2 = self.summary2[0:2]
        
        ProfileUtils.profile('metrics.distance(self.summary1, self.summary2)', globals(), locals())


profiler = HIVGraphMetricsProfile()
profiler.profileDistance()
#

