"""
Use the DBLP dataset to recommend experts. Find the optimal parameters. 
"""
import gc 
import os
import numpy 
import logging 
import sys 
import argparse
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 
from apgl.util.Evaluator import Evaluator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3, linewidth=160)
numpy.random.seed(21)

parser = argparse.ArgumentParser(description='Run reputation evaluation experiments')
parser.add_argument("-r", "--runLDA", action="store_true", help="Run Latent Dirchlet Allocation")
args = parser.parse_args()

averagePrecisionN = 50 
similarityCutoff = 0.30
ns = numpy.arange(5, 105, 5)
runLSI = not args.runLDA

dataset = ArnetMinerDataset(runLSI=runLSI) 
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-5000000.txt"
dataset.dataFilename = dataset.dataDir + "DBLP-citation-7000000.txt"
dataset.ks = [100, 150, 200, 250, 300, 350]
dataset.overwriteGraph = True
dataset.overwriteModel = True

dataset.overwriteVectoriser = True 
dataset.vectoriseDocuments()
dataset.overwriteVectoriser = False

dataset.modelSelection()
dataset.overwriteModel = False

for field in dataset.fields: 
    logging.debug("Field = " + field)
    dataset.learnModel() 
    relevantExperts = dataset.findSimilarDocuments(field)
    
    graph, authorIndexer = dataset.coauthorsGraph(field, relevantExperts)
    expertMatches = dataset.matchExperts(relevantExperts, dataset.testExpertDict[field])     
    
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
        
        precisions2 = numpy.c_[numpy.array(ns), precisions]
        
        logging.debug(Latex.array2DToRows(precisions2))
        logging.debug(Latex.array1DToRow(averagePrecisions))
    
        resultsFilename = dataset.getResultsDir(field) + "precisions.npz"
        numpy.savez(resultsFilename, precisions, averagePrecisions)
        logging.debug("Saved precisions as " + resultsFilename)

logging.debug("All done!")
