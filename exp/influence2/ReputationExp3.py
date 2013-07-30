"""
Use the new dataset from Pierre to perform expert recommendation. 
"""
import numpy 
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
import xml.etree.ElementTree as ET
import logging 
import sys 
import array 
import itertools
import difflib 
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Evaluator import Evaluator 
from exp.util.IdIndexer import IdIndexer 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator
from apgl.util.Latex import Latex 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

field = "Boosting"
#field = "MachineLearning"
dirName = PathDefaults.getDataDir() + "reputation/" + field + "/" 

coauthorFilename = dirName + field.lower() + ".csv"
trainExpertsFilename = dirName + field.lower() + "_seed_train" + ".csv"
testExpertsFilename = dirName + field.lower() + "_seed_test" + ".csv"

#This is a list of experts that match authors as written in the coauthors file 
expertsFilename = dirName + "experts.txt"
coauthorFile = open(coauthorFilename)

authorIndexer = IdIndexer("i")
articleIndexer = IdIndexer("i")

for line in coauthorFile: 
    vals = line.split(";")
    
    authorId = vals[0].strip().strip("=")
    articleId = vals[1].strip()
    
    authorIndexer.append(authorId)
    articleIndexer.append(articleId)

authorInds = authorIndexer.getArray()
articleInds = articleIndexer.getArray()
edges = numpy.c_[authorInds, articleInds]

author1Inds = array.array('i')
author2Inds = array.array('i')

lastArticleInd = -1
coauthorList = []

#Go through and create coauthor graph 
for i in range(authorInds.shape[0]): 
    authorInd = authorInds[i]    
    articleInd = articleInds[i] 
    
    if articleInd != lastArticleInd:         
        iterator = itertools.combinations(coauthorList, 2)
        for vId1, vId2 in iterator:   
            author1Inds.append(vId1)
            author2Inds.append(vId2)
        
        coauthorList = []
        coauthorList.append(authorInd)
    else: 
        coauthorList.append(authorInd)
        
    lastArticleInd = articleInd

author1Inds = numpy.array(author1Inds, numpy.int)
author2Inds = numpy.array(author2Inds, numpy.int)
edges = numpy.c_[author1Inds, author2Inds]

#Coauthor graph is undirected 
graph = igraph.Graph()
graph.add_vertices(numpy.max(authorInds)+1)
graph.add_edges(edges)

print(graph.summary())

logging.debug("Number of components in graph: " + str(len(graph.components()))) 
compSizes = [len(x) for x in graph.components()]
logging.debug("Max component size: " + str(numpy.max(compSizes))) 

computeInfluence = False
outputLists = GraphRanker.rankedLists(graph, numRuns=100, computeInfluence=computeInfluence, p=0.01)
itemList = RankAggregator.generateItemList(outputLists)
methodNames = GraphRanker.getNames(computeInfluence=computeInfluence)

#outputList, scores = RankAggregator.MC2(outputLists, itemList)
#outputLists.append(outputList)
#methodNames.append("MC2")

#Now load list of experts
expertsFile = open(testExpertsFilename)
expertsList = []
expertsIdList = []  
i = 0

for line in expertsFile: 
    vals = line.split() 
    key = vals[0][0].lower() + "/" + vals[0].strip(",") + ":" 
    expertName = vals[-1] + ", "    
    
    for j in range(1, len(vals)): 
        if j != len(vals)-2:
            key += vals[j].strip(".,") + "_"
        else: 
            key += vals[j].strip(".,")
            
        expertName += vals[j] + " "
        
    key = key.strip()
    expertName = expertName.strip()
        
    possibleExperts = difflib.get_close_matches(key, authorIndexer.getIdDict().keys(), cutoff=0.8)        
        
    if len(possibleExperts) != 0:
        #logging.debug("Matched key : " + key + ", " + possibleExperts[0])
        expertsIdList.append(authorIndexer.getIdDict()[possibleExperts[0]])
        expertsList.append(expertName) 
        
    else: 
        logging.debug("Key not found : " + line.strip() + ": " + key)
        possibleExperts = difflib.get_close_matches(key, authorIndexer.getIdDict().keys(), cutoff=0.6) 
        logging.debug("Possible matches: " + str(possibleExperts))
        
    i += 1 


expertsFile.close()
logging.debug("Found " + str(len(expertsIdList)) + " of " + str(i) + " experts")

#outputList2, scores2 = RankAggregator.supervisedMC22(outputLists, itemList, expertsList)
#outputLists.append(outputList2)
#methodNames.append("SMC2")

print(outputLists[0][0:10])
for ind in outputLists[0][0:10]: 
    key = (key for key,value in authorIndexer.getIdDict().items() if value==ind).next()
    print(key)

ns = numpy.arange(5, 105, 5)
numMethods = len(outputLists)
precisions = numpy.zeros((len(ns), numMethods))

for i, n in enumerate(ns):     
    for j in range(len(outputLists)): 
        precisions[i, j] = Evaluator.precision(expertsIdList, outputLists[j][0:n])

precisions = numpy.c_[numpy.array(ns), precisions]
    
    
print(Latex.latexTable(Latex.array2DToRows(precisions), colNames=methodNames))
