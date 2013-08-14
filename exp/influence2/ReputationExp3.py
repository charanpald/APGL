"""
Use the DBLP dataset to recommend experts. Find the optimal parameters. 
"""
import gc 
import os
import numpy 
import logging 
import sys 
import sklearn.metrics 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment"]
#fields = ["Boosting"]
ks = [50, 100, 150]
#ks = [50]
maxRelevantAuthors = [100, 200, 500, 1000]
#maxRelevantAuthors = [100, 200]
similarityCutoffs = [0.2, 0.3, 0.4, 0.5]
#similarityCutoffs = [0.2]
bestAveragePrecision = numpy.zeros((len(fields), len(maxRelevantAuthors), len(similarityCutoffs)))
ns = numpy.arange(5, 105, 5)

for r, field in enumerate(fields): 
    dataset = ArnetMinerDataset(field, k=50)    
    for s, maxRelAuthors in enumerate(maxRelevantAuthors): 
        for t, similarityCutoff in enumerate(similarityCutoffs): 
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
            outputLists = GraphRanker.rankedLists(graph, numRuns=100, computeInfluence=computeInfluence, p=0.05, trainExpertsIdList=expertMatchesInds)
            itemList = RankAggregator.generateItemList(outputLists)
            #methodNames = GraphRanker.getNames(computeInfluence=computeInfluence)
            outputLists.append(relevantExperts)
            
            #Process outputLists to only include people from the relevant field  
            newOutputLists = []
            for lst in outputLists: 
                lst = lst[lst < len(relevantAuthorInds)]  
                newOutputLists.append(lst)
            
            print("\n")
            
            numMethods = len(newOutputLists)
            precisions = numpy.zeros((len(ns), numMethods))
            averagePrecisions = numpy.zeros(numMethods)
            
            for i, n in enumerate(ns):     
                for j in range(len(outputLists)): 
                    predY = -numpy.ones(len(relevantAuthorInds))
                    predY[expertMatchesInds] = 1
                    
                    testY = -numpy.ones(len(relevantAuthorInds))
                    testY[newOutputLists[j][0:n]] = 1
                    
                    precisions[i, j] = sklearn.metrics.precision_score(testY, predY) 
        
            n = 50 
            
            for j in range(len(outputLists)): 
                predY = -numpy.ones(len(relevantAuthorInds))
                predY[expertMatchesInds] = 1
                
                testY = -numpy.ones(len(relevantAuthorInds))
                testY[newOutputLists[j][0:n]] = 1
                
                averagePrecisions[j] = sklearn.metrics.average_precision_score(testY, predY)
            
            precisions = numpy.c_[numpy.array(ns), precisions]
            
            logging.debug(Latex.array1DToRow(averagePrecisions*len(expertMatches)))
            bestAveragePrecision[r,s,t] = numpy.max(averagePrecisions)*len(expertMatches) 
            
            logging.debug("Max average precision for " + str((field, maxRelAuthors, similarityCutoff)) + " = " + str(bestAveragePrecision[r,s,t]))

            
for r in range(bestAveragePrecision.shape[0]):
    print("field = " + str(fields[r]))
    print(bestAveragePrecision[r, :, :])

logging.debug("All done!")