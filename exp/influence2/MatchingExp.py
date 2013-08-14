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

ks = numpy.array([50, 75, 100])
maxRelevantAuthors = numpy.array([100, 200, 500]) 
fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment"]

coverage1 = numpy.zeros((len(ks), len(fields), maxRelevantAuthors.shape[0]))
coverage2 = numpy.zeros((len(ks), len(fields), maxRelevantAuthors.shape[0]))


for m, k in enumerate(ks): 
    overwriteSVD = True
    for i, field in enumerate(fields): 
        for j, maxRelAuthors in enumerate(maxRelevantAuthors): 
            logging.debug("k=" + str(k) + " field=" + field + " maxRelAuthors=" + str(maxRelAuthors))
            dataset = ArnetMinerDataset(field)
            dataset.k = k 
            dataset.overwriteRelevantExperts = True
            dataset.overwriteCoauthors = True
            dataset.overwriteSVD = overwriteSVD
            dataset.maxRelevantAuthors = maxRelAuthors
            
            dataset.vectoriseDocuments()
            dataset.findSimilarDocuments()
    
            graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
            expertMatches, expertsSet = dataset.matchExperts()
            
            coverage1[m, i, j] = float(len(expertMatches))/len(relevantExperts)
            coverage2[m, i, j] = float(len(expertMatches))/len(expertsSet)
            
            overwriteSVD = False
            
            logging.debug("Coverage 2: " + str((k, field, maxRelAuthors)) + " = " + str(coverage2[m, i, j]))

for m, k in enumerate(ks): 
    print("k="+str(k))      
    print(coverage1[m, :, :])
    print(coverage2[m, :, :])

resultsFilename = PathDefaults.getOutputDir() + "MatchingResults.npz"
numpy.savez(resultsFilename, coverage1, coverage2)
