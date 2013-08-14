"""
Use the DBLP dataset to recommend experts. Find the optimal parameters. 
"""
import gc 
import os
import numpy 
import logging 
import sys 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 
from apgl.util.Evaluator import Evaluator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3, linewidth=160)
numpy.random.seed(21)

averagePrecisionN = 50 
fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment"]
#fields = ["Boosting"]
similarityCutoff = 0.2
k = 100
maxRelevantAuthors = [100, 200, 500, 1000]
#maxRelevantAuthors = [100, 200]
bestAveragePrecision = numpy.zeros((len(fields), len(maxRelevantAuthors)))
ns = numpy.arange(5, 105, 5)

for r, field in enumerate(fields): 
    dataset = ArnetMinerDataset(field, k=k)    
    for s, maxRelAuthors in enumerate(maxRelevantAuthors): 
        dataset.overwriteRelevantExperts = True
        dataset.overwriteCoauthors = True
        dataset.maxRelevantAuthors = maxRelAuthors
        dataset.similarityCutoff = similarityCutoff
        
        dataset.vectoriseDocuments()
        dataset.findSimilarDocuments()
        
        graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
        expertMatches, expertsSet = dataset.matchExperts()     
    
        expertMatchesInds = authorIndexer.translate(expertMatches) 
        relevantAuthorInds = authorIndexer.translate(relevantExperts) 
        assert (numpy.array(relevantAuthorInds) < len(relevantAuthorInds)).all()
        
        #First compute graph properties 
        computeInfluence = False
        graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, trainExpertsIdList=expertMatchesInds)
        outputLists = graphRanker.vertexRankings(graph, relevantAuthorInds, [relevantAuthorInds])
        itemList = RankAggregator.generateItemList(outputLists)
        #methodNames = graphRanker.getNames()
        
        numMethods = len(outputLists)
        precisions = numpy.zeros((len(ns), numMethods))
        averagePrecisions = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                precisions[i, j] = Evaluator.precisionFromIndLists(expertMatchesInds, outputLists[j][0:n]) 
                
        for j in range(len(outputLists)):                 
            averagePrecisions[j] = Evaluator.averagePrecisionFromLists(expertMatchesInds, outputLists[j][0:averagePrecisionN], averagePrecisionN) 
        
        precisions = numpy.c_[numpy.array(ns), precisions]
        logging.debug(Latex.array1DToRow(averagePrecisions*len(expertMatches)))
        bestAveragePrecision[r,s] = numpy.max(averagePrecisions)*len(expertMatches) 
        logging.debug("Max average precision for " + str((field, maxRelAuthors)) + " = " + str(bestAveragePrecision[r,s]))

for r in range(bestAveragePrecision.shape[0]):
    print("field = " + str(fields[r]))
    print(bestAveragePrecision[r, :])

logging.debug("All done!")
