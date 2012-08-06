import numpy 
import logging 

from apgl.util.Parameter import Parameter
from exp.viroscopy.model.HIVGraph import HIVGraph 
from exp.sandbox.GraphMatch import GraphMatch 

class HIVGraphMetrics2(object): 
    def __init__(self, realGraph, epsilon, matcher=None, numTimeSteps=10):
        """
        A class to model metrics about and between HIVGraphs such as summary 
        statistics and distances. In this case we perform graph matching 
        using the PATH algorithm and other graph matching methods. 
        
        :param times: An array of time points to compute statistics from 
        """
        
        self.dists = [] 
        self.realGraph = realGraph
        self.epsilon = epsilon 
        self.numTimeSteps = numTimeSteps 
        
        if matcher == None: 
            self.matcher = GraphMatch("U")
        else: 
            self.matcher = matcher 
        
    def addGraph(self, graph): 
        """
        Compute the distance between this graph and the realGraph at the time 
        of the last event of this one. 
        """
        t = graph.endTime()
        subgraph = graph.subgraph(graph.removedIndsAt(t))  
        subRealGraph = self.realGraph.subgraph(self.realGraph.removedIndsAt(t))  
        permutation, distance, time = self.matcher.match(subgraph, subRealGraph)
        lastDist = self.matcher.distance(subgraph, subRealGraph, permutation, True, True) 
        
        logging.debug("Distance at time " + str(t) + " is " + str(lastDist) + " with simulated size " + str(subgraph.size) + " and real size " + str(subRealGraph.size))        
        
        self.dists.append(lastDist)
    
    def distance(self): 
        """
        If we have the required number of time steps, return the mean distance 
        otherwise return a distance of 1 (the max distance).
        """
        if self.numTimeSteps == len(self.dists): 
            return self.meanDistance()
        else: 
            return 1 
        
    def meanDistance(self):
        """
        This is the mean distance of the graph matches so far. 
        """
        dists = numpy.array(self.dists)
        if dists.shape[0]!=0: 
            return dists.mean()
        else: 
            return 1
        
    def shouldBreak(self): 
        return self.meanDistance() > self.epsilon 