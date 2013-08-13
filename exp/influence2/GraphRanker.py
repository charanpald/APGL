import numpy 
import logging
from exp.influence2.MaxInfluence import MaxInfluence 

class GraphRanker(object): 
    def __init__(self): 
        pass 

    @staticmethod
    def getNames(computeInfluence=False): 
        names = ["Betweenness", "Closeness", "PageRank", "Degree"]
        
        if computeInfluence: 
            names.append("Influence")
        
        names.append("Shortest path")
        
        return names 

    @staticmethod     
    def rankedLists(graph, k=100, p=0.5, numRuns=1000, computeInfluence=False, trainExpertsIdList=None): 
        """
        Return a list of ranked lists. The list is: betweenness, pagerank, 
        degree and influence. 
        """
        outputLists = []
        
        logging.debug("Computing betweenness")
        scores = graph.betweenness(weights="invWeight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing closeness")
        scores = graph.closeness(weights="invWeight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing PageRank")
        scores = graph.pagerank(weights="weight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing weighted degree distribution")
        #scores = graph.degree(graph.vs)
        scores = graph.strength(weights="weight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        if computeInfluence: 
            logging.debug("Computing influence")
            rank = MaxInfluence.greedyMethod2(graph, k, p=p, numRuns=numRuns)
            outputLists.append(numpy.array(rank))
        
        """
        logging.debug("Computing shortest path lengths")
        lengths = graph.shortest_paths(trainExpertsIdList, weights="invWeight")
        lengths = numpy.array(lengths)
        lengths[numpy.logical_not(numpy.isfinite(lengths))] = 0
        lengths = numpy.mean(lengths, 0)
        rank = numpy.argsort(lengths)
        outputLists.append(rank)
        """
        
        logging.debug("Computing hub score")
        scores = graph.hub_score(weights="weight") 
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        logging.debug("Computing authority score")
        scores = graph.authority_score(weights="weight") 
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(rank)
        
        return outputLists 
