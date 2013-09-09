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
from apgl.util.Util import Util

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3, linewidth=160)
numpy.random.seed(21)

parser = argparse.ArgumentParser(description='Run reputation evaluation experiments')
parser.add_argument("-r", "--runLDA", action="store_true", help="Run Latent Dirchlet Allocation")
args = parser.parse_args()

averagePrecisionN = 20 
ns = numpy.arange(5, 55, 5)
runLSI = not args.runLDA

dataset = ArnetMinerDataset(runLSI=runLSI) 
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-1000000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-1000000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-5000000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-7000000.txt"
dataset.dataFilename = dataset.dataDir + "DBLP-citation-Feb21.txt" 
dataset.minDf = 10**-2
dataset.ks = [100, 200, 300, 400, 500, 600]
dataset.minDfs = [10**-4]
dataset.overwriteGraph = True
dataset.overwriteModel = True
dataset.overwriteVectoriser = True 

#dataset.modelSelection()

for field in dataset.fields: 
    logging.debug("Field = " + field)
    dataset.learnModel() 
    dataset.overwriteVectoriser = False
    dataset.overwriteModel = False    
    
    relevantAuthors = dataset.findSimilarDocuments(field)
    
    graph, authorIndexer = dataset.coauthorsGraph(field, relevantAuthors)
    trainExpertMatches = dataset.matchExperts(relevantAuthors, dataset.trainExpertDict[field])   
    testExpertMatches = dataset.matchExperts(relevantAuthors, dataset.testExpertDict[field])     
    
    trainExpertMatchesInds = authorIndexer.translate(trainExpertMatches)
    testExpertMatchesInds = authorIndexer.translate(testExpertMatches) 
    relevantAuthorInds = authorIndexer.translate(relevantAuthors) 
    assert (numpy.array(relevantAuthorInds) < len(relevantAuthorInds)).all()
    
    if len(testExpertMatches) != 0: 
        #First compute graph properties 
        computeInfluence = True
        graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, inputRanking=relevantAuthorInds)
        outputLists = graphRanker.vertexRankings(graph, relevantAuthorInds)
        itemList = RankAggregator.generateItemList(outputLists)
        #methodNames = graphRanker.getNames()
        
        if runLSI: 
            outputFilename = dataset.getOutputFieldDir(field) + "outputListsLSI.npz"
        else: 
            outputFilename = dataset.getOutputFieldDir(field) + "outputListsLDA.npz"
            
        Util.savePickle([outputLists, trainExpertMatchesInds, testExpertMatchesInds], outputFilename, debug=True)
        
        numMethods = len(outputLists)
        precisions = numpy.zeros((len(ns), numMethods))
        averagePrecisions = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                precisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, outputLists[j][0:n]) 
            
        for j in range(len(outputLists)):                 
            averagePrecisions[j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, outputLists[j][0:averagePrecisionN], averagePrecisionN) 
        
        precisions2 = numpy.c_[numpy.array(ns), precisions]
        
        logging.debug(Latex.array2DToRows(precisions2))
        logging.debug(Latex.array1DToRow(averagePrecisions))

logging.debug("All done!")
