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
similarityCutoff = 0.30
ns = numpy.arange(5, 105, 5)

dataset = ArnetMinerDataset() 
dataset.dataFilename = dataset.dataDir + "DBLP-citation-1000000.txt"
dataset.overwrite = True
dataset.overwriteModel = True
dataset.overwriteVectoriser = True 

dataset.modelSelectionLSI()

for field in dataset.fields: 
    logging.debug("Field = " + field)
    relevantExperts = dataset.findSimilarDocumentsLSI(field)
    
    graph, authorIndexer = dataset.coauthorsGraph(field, relevantExperts)
    expertMatches = dataset.matchExperts(relevantExperts, dataset.testExpertDict)     
    
    expertMatchesInds = authorIndexer.translate(expertMatches) 
    relevantAuthorInds = authorIndexer.translate(relevantExperts) 
    assert (numpy.array(relevantAuthorInds) < len(relevantAuthorInds)).all()
    
    if len(expertMatches) != 0: 
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
        
        logging.debug(Latex.array2DToRow2(precisions*len(expertMatches)))
        logging.debug(Latex.array1DToRow(averagePrecisions*len(expertMatches)))
    
        resultsFilename = dataset.getResultsDir(field) + "precisions.npz"
        numpy.savez(resultsFilename, precisions, averagePrecisions)

logging.debug("All done!")
