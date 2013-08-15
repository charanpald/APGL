"""
Let's tune parameters to capture as many of the authors in the relevant field 
as possible. 
"""

import numpy 
import logging 
import sys 
import argparse 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.PathDefaults import PathDefaults

parser = argparse.ArgumentParser()
parser.add_argument('--runLDA', action='store_true')
args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

ks = numpy.array([100, 150, 200])
maxRelevantAuthors = numpy.array([100, 200, 500]) 
fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment"]

coverage1 = numpy.zeros((len(ks), len(fields), maxRelevantAuthors.shape[0]))
coverage2 = numpy.zeros((len(ks), len(fields), maxRelevantAuthors.shape[0]))
runLDA = args.runLDA
print(runLDA)

for m, k in enumerate(ks): 
    overwriteVectoriser = True
    overwriteModel = True
    
    for i, field in enumerate(fields): 
        for j, maxRelAuthors in enumerate(maxRelevantAuthors): 
            logging.debug("k=" + str(k) + " field=" + field + " maxRelAuthors=" + str(maxRelAuthors))
            dataset = ArnetMinerDataset(field)
            dataset.k = k 
            dataset.overwrite = True
            dataset.overwriteVectoriser = overwriteVectoriser
            dataset.overwriteModel = overwriteModel
            dataset.maxRelevantAuthors = maxRelAuthors
            dataset.dataFilename = dataset.dataDir + "DBLP-citation-1000000.txt"
            
            if runLDA:
                dataset.findSimilarDocumentsLDA()
            else: 
                dataset.findSimilarDocumentsLSI()
    
            graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
            expertMatches, expertsSet = dataset.matchExperts()
            
            coverage1[m, i, j] = float(len(expertMatches))/len(relevantExperts)
            coverage2[m, i, j] = float(len(expertMatches))/len(expertsSet)
            
            overwriteVectoriser = False 
            overwriteModel = False
            
            logging.debug("Coverage 2: " + str((k, field, maxRelAuthors)) + " = " + str(coverage2[m, i, j]))

for m, k in enumerate(ks): 
    print("k="+str(k))      
    print(coverage1[m, :, :])
    print(coverage2[m, :, :])

if runLDA: 
    resultsFilename = PathDefaults.getOutputDir() + "MatchingResultsLDA.npz"
else: 
    resultsFilename = PathDefaults.getOutputDir() + "MatchingResultsLSI.npz"
numpy.savez(resultsFilename, coverage1, coverage2)
