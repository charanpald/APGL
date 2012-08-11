import numpy 
import logging 

from apgl.util.Parameter import Parameter
from exp.viroscopy.model.HIVGraph import HIVGraph 
from exp.sandbox.GraphMatch import GraphMatch 

class HIVGraphMetrics2(object): 
    def __init__(self, realGraph, epsilon, matcher=None, T=1000):
        """
        A class to model metrics about and between HIVGraphs such as summary 
        statistics and distances. In this case we perform graph matching 
        using the PATH algorithm and other graph matching methods. 
        
        :param realGraph: The target epidemic graph 
        
        :param epsilon: The max mean distance before we break the simulation
        
        :param matcher: The graph matcher object to compute graph distances. 
        
        :param T: The end time of the simulation. If the simulation quits before T, then distance = 1.
        """
        
        self.dists = [] 
        self.realGraph = realGraph
        self.epsilon = epsilon 
        self.T = T 
        self.times = []
        
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
        
        #Only add distance if the real graph has nonzero size
        if subRealGraph.size != 0: 
            permutation, distance, time = self.matcher.match(subgraph, subRealGraph)
            lastDist = self.matcher.distance(subgraph, subRealGraph, permutation, True, True) 
            
            logging.debug("Distance at time " + str(t) + " is " + str(lastDist) + " with simulated size " + str(subgraph.size) + " and real size " + str(subRealGraph.size))        
            
            self.dists.append(lastDist)
            self.times.append(t) 
        else: 
            logging.debug("Not adding distance at time " + str(t) + " with simulated size " + str(subgraph.size) + " and real size " + str(subRealGraph.size))
    
    def distance(self): 
        """
        If we have the required number of time steps, return the mean distance 
        otherwise return a distance of 1 (the max distance).
        """
        if len(self.times) != 0 and self.times[-1] >= self.T: 
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
            return 0
        
    def shouldBreak(self): 
        return self.meanDistance() > self.epsilon 
        