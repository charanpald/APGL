import numpy 
import logging
from exp.influence2.MaxInfluence import MaxInfluence 
from exp.influence2.RankAggregator import RankAggregator

class GraphRanker(object): 
    def __init__(self, k=100, p=0.5, numRuns=1000, computeInfluence=False, inputRanking=None): 
        self.k = k 
        self.p = p 
        self.numRuns = numRuns
        self.computeInfluence = computeInfluence 
        self.inputRanking = inputRanking
        
    def getNames(self): 
        if self.inputRanking == None: 
            names = ["Betweenness", "Closeness", "PageRank", "Degree"]
        else: 
            names = ["InputRanking", "Betweenness", "Closeness", "PageRank", "Degree"]
        
        if self.computeInfluence: 
            names.append("Influence")
        
        names.append("Hub score")
        
        return names 
 
    def vertexRankings(self, graph, relevantItems): 
        """
        Return a list of ranked lists. The list is: betweenness, pagerank, 
        degree and influence. 
        """
        if self.inputRanking == None: 
            outputLists = []
        else: 
            outputLists = [self.inputRanking]
        
        logging.debug("Computing betweenness")
        scores = graph.betweenness(weights="invWeight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(self.restrictRankedList(rank, relevantItems)) 
        
        logging.debug("Computing closeness")
        scores = graph.closeness(weights="invWeight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(self.restrictRankedList(rank, relevantItems))
        
        logging.debug("Computing PageRank")
        scores = graph.pagerank(weights="weight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(self.restrictRankedList(rank, relevantItems))
        
        logging.debug("Computing weighted degree distribution")
        scores = graph.strength(weights="weight")
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(self.restrictRankedList(rank, relevantItems))
        
        if self.computeInfluence: 
            logging.debug("Computing influence")
            rank = MaxInfluence.greedyMethod2(graph, self.k, p=self.p, numRuns=self.numRuns)
            outputLists.append(numpy.array(self.restrictRankedList(rank, relevantItems)))
        
        logging.debug("Computing hub score")
        scores = graph.hub_score(weights="weight") 
        rank = numpy.flipud(numpy.argsort(scores)) 
        outputLists.append(self.restrictRankedList(rank, relevantItems))
        
        #Now add MC2 aggregated rankings 
        #logging.debug("Computing MC2 rank aggregation")
        #rank = RankAggregator.MC2(outputLists, relevantItems)[0]
        #outputLists.append(rank)
        
        return outputLists 

    def restrictRankedList(self, lst, releventList):
        """
        Given an ordered list lst, restrict items to those in releventList, 
        retaining the order. 
        """
        releventList = set(releventList)
        newList = [] 
        for item in lst: 
            if item in releventList: 
                newList.append(item)
        return newList 
