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
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Evaluator import Evaluator 
from exp.util.IdIndexer import IdIndexer 
from exp.influence2.GraphRanker import GraphRanker 
from exp.influence2.RankAggregator import RankAggregator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dirName = PathDefaults.getDataDir() + "reputation/IntelligentAgents/" 
#dirName = PathDefaults.getDataDir() + "reputation/OntologyAlignment/" 

coauthorFilename = dirName + "articles.csv"
expertsFilename = dirName + "experts.txt"
coauthorFile = open(coauthorFilename)

authorIndexer = IdIndexer("i")
articleIndexer = IdIndexer("i")

for line in coauthorFile: 
    vals = line.split(";")
    
    authorId = vals[0].strip()
    if "_" in authorId: 
        authorId = authorId[0:authorId.find("_")]    
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

outputLists = GraphRanker.rankedLists(graph, computeInfluence=True)
itemList = RankAggregator.generateItemList(outputLists)
outputList, scores = RankAggregator.MC2(outputLists, itemList)

#Now load list of experts
expertsFile = open(expertsFilename)
expertsList = [] 
i = 0

for line in expertsFile: 
    vals = line.split() 
    key = vals[1][0].lower() + "/" + vals[1] + ":" + vals[0]
    
    if key in authorIndexer.getIdDict(): 
        expertsList.append(authorIndexer.getIdDict()[key])
        
    i += 1 

expertsFile.close()
logging.debug("Found " + str(len(expertsList)) + " of " + str(i) + " experts")

ns = [5, 10, 15, 20, 25, 30, 35]
numMethods = 1+len(outputLists)
precisions = numpy.zeros((len(ns), numMethods))

for i, n in enumerate(ns): 
    precisions[i, 0] = Evaluator.precision(expertsList, outputList[0:n])
    
    for j in range(len(outputLists)): 
        precisions[i, j+1] = Evaluator.precision(expertsList, outputLists[j][0:n])
    
print(precisions)
