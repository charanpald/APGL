"""
Use the DBLP dataset to recommend experts. 
"""
import numpy 
import logging 
import sys 
from apgl.util.Evaluator import Evaluator 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from exp.influence2.GraphReader2 import GraphReader2
from apgl.util.Latex import Latex 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

field = "Boosting" 
#field = "IntelligentAgents"
#field = "MachineLearning"

dataset = ArnetMinerDataset(field)
dataset.vectoriseDocuments()
dataset.findSimilarDocuments()

graph, authorIndexer, relevantExperts = dataset.coauthorsGraph()
expertMatches = dataset.matchExperts()

print(graph.summary())

logging.debug(expertMatches)

expertMatchesInds = [] 
for expert in expertMatches: 
    expertMatchesInds.append(authorIndexer.translate(expert))
   
logging.debug(expertMatchesInds)   
   
relevantAuthorInds = [] 
for author in relevantExperts: 
    relevantAuthorInds.append(authorIndexer.translate(author))
    

#First compute graph properties 
computeInfluence = False
outputLists = GraphRanker.rankedLists(graph, numRuns=100, computeInfluence=computeInfluence, p=0.05, trainExpertsIdList=expertMatchesInds)
itemList = RankAggregator.generateItemList(outputLists)
methodNames = GraphRanker.getNames(computeInfluence=computeInfluence)

#Then use MC2 rank aggregation 
#outputList, scores = RankAggregator.MC2(outputLists, itemList)
#outputLists.append(outputList)
#methodNames.append("MC2")
#print("outputList="+str(outputList))

#The supervised MC2
#outputList2, scores2 = RankAggregator.supervisedMC22(outputLists, itemList, expertsIdList)
#outputLists.append(outputList2)
#methodNames.append("SMC2")

#Process outputLists to only include people from the relevant field  
newOutputLists = []
for lst in outputLists: 
    print(len(lst))
    lst = lst[lst <= len(relevantAuthorInds)]  
    newOutputLists.append(lst)

print("\n")
"""
r = 20 
logging.debug("Top " + str(r) + " authors:")
for ind in outputLists[-1][0:r]: 
    key = (key for key,value in reader.authorIndexer.getIdDict().items() if value==ind).next()
    logging.debug(key)
"""

print("\n")

ns = numpy.arange(5, 105, 5)
numMethods = len(newOutputLists)
precisions = numpy.zeros((len(ns), numMethods))

for i, n in enumerate(ns):     
    for j in range(len(outputLists)): 
        precisions[i, j] = Evaluator.precision(expertMatchesInds, newOutputLists[j][0:n])

precisions = numpy.c_[numpy.array(ns), precisions]
print(Latex.latexTable(Latex.array2DToRows(precisions), colNames=methodNames))
