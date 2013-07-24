import numpy 
import logging
from exp.influence2.MaxInfluence import MaxInfluence 

class GraphRanker(object): 
    def __init__(self): 
        pass 


    @staticmethod     
    def rankedLists(graph, k=100, p=0.5): 
        """
        Return a list of ranked lists. The list is: betweenness, pagerank, 
        degree and influence. 
        """
        outputLists = []
        
        logging.debug("Computing betweenness")
        scores = graph.betweenness()
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing PageRank")
        scores = graph.pagerank()
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing degree distribution")
        scores = graph.degree(graph.vs)
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        #logging.debug("Computing influence")
        #rank = MaxInfluence.celf(graph, k, p=p)
        #outputLists.append(numpy.array(rank))
        
        return outputLists 
