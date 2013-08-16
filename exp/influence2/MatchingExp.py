"""
Let's tune parameters to capture as many of the authors in the relevant field 
as possible. 
"""

import numpy 
import logging 
import sys 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.PathDefaults import PathDefaults

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

algorithms = ["lsi", "lsi2", "lda"]
useTfidf = [True, False]
#ks = numpy.array([100, 150, 200])
ks = numpy.array([50])
#maxRelevantAuthors = numpy.array([100, 200, 500]) 
maxRelevantAuthors = numpy.array([100]) 
fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment"]

coverage1 = numpy.zeros((len(useTfidf), len(algorithms), len(ks), len(fields), maxRelevantAuthors.shape[0]))
coverage2 = numpy.zeros((len(useTfidf), len(algorithms), len(ks), len(fields), maxRelevantAuthors.shape[0]))

for p, tfidf in enumerate(useTfidf):
    for n, algorithm in enumerate(algorithms):
        for m, k in enumerate(ks): 
            overwriteVectoriser = True
            overwriteModel = True
                    
            for i, field in enumerate(fields): 
                for j, maxRelAuthors in enumerate(maxRelevantAuthors): 
                    logging.debug("k=" + str(k) + " field=" + field + " maxRelAuthors=" + str(maxRelAuthors) + " algorithm=" + str(algorithm) + " useTfidf=" + str(tfidf))
                    
                    dataset = ArnetMinerDataset(field)
                    dataset.k = k 
                    dataset.overwrite = True
                    dataset.overwriteVectoriser = overwriteVectoriser
                    dataset.overwriteModel = overwriteModel
                    dataset.maxRelevantAuthors = maxRelAuthors
                    dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt"
                    dataset.tfidf = tfidf
                    
                    if algorithm == "lsi":
                        dataset.findSimilarDocumentsLSI()
                    elif algorithm == "lsi2": 
                        dataset.findSimilarDocumentsLSI2()
                    else: 
                        dataset.findSimilarDocumentsLDA()
            
                    graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
                    expertMatches, expertsSet = dataset.matchExperts()
                    
                    if len(relevantExperts) != 0: 
                        coverage1[p, n, m, i, j] = float(len(expertMatches))/len(relevantExperts)
                    if len(expertsSet) != 0: 
                        coverage2[p, n, m, i, j] = float(len(expertMatches))/len(expertsSet)
                    
                    overwriteVectoriser = False 
                    overwriteModel = False
                    
                    logging.debug("Coverage 2: " + str((tfidf, algorithm, k, field, maxRelAuthors)) + " = " + str(coverage2[p, n, m, i, j]))

meanCoverage1 = numpy.mean(coverage1, 3)
meanCoverage2 = numpy.mean(coverage2, 3)

for p, tfidf in enumerate(useTfidf):
    for n, algorithm in enumerate(algorithms):
        for m, k in enumerate(ks): 
            logging.debug(str((tfidf, algorithm, k)) + " " + str(meanCoverage2[p, n, m, :]))


resultsFilename = PathDefaults.getOutputDir() + "MatchingResults.npz"
numpy.savez(resultsFilename, coverage1, coverage2)
