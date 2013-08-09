"""
Let's tune parameters to capture as many of the authors in the relevant field 
as possible. 
"""

import numpy 
import logging 
import sys 
from apgl.util.Evaluator import Evaluator 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

fields = ["Boosting", "Intelligent Agents", "Machine Learning", "Ontology Alignment", "Neural Networks" ]
cutoffs = numpy.array([0.3, 0.4, 0.5, 0.6, 0.7])  
coverage = numpy.zeros((len(fields), cutoffs.shape[0]))


for i, field in enumerate(fields): 
    for j, cutoff in enumerate(cutoffs): 
        logging.debug("field=" + field + " cutoff=" + str(cutoff))
        dataset = ArnetMinerDataset(field)
        dataset.overwriteRelevantExperts = True
        dataset.overwriteCoauthors = True
        dataset.similarityCutoff = cutoff
        
        dataset.vectoriseDocuments()
        dataset.findSimilarDocuments()

        graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
        expertMatches = dataset.matchExperts()
        
        coverage[i, j] = float(len(expertMatches))/len(relevantExperts)
        
print(coverage)

print(numpy.mean(coverage, 0))
