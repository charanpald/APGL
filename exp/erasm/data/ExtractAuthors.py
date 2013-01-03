
"""
We will analyse the article metadata file and extract co-authors. 
"""

from apgl.util.PathDefaults import PathDefaults 
import logging 
import sys 
import itertools 
import igraph 
import matplotlib.pyplot as plt 
import json 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#path = "/local/dhanjalc/dataDump-28-11-12/" 
path = PathDefaults.getDataDir() + "erasm/"

#fileName = path + "articleMetadata-28-11-12"
fileName = path + "articleMetadata100000"


fileObj = open(fileName, 'r')
articleMetadataList = []
vertexIdDict = {} 
vertexIdSet = set([])
vertexIdList = []
edgeSet = set([])
edgeArray = []

i = 0 

for line in fileObj: 
    if i % 1000 == 0: 
        print(i)
    
    articleMetaData = json.loads(line)
    
    if "authors" in articleMetaData: 
        authors = articleMetaData["authors"]
        
        coauthorList = []
        for author in authors: 
            authorString = author["forename"] + " " + author["surname"]
            authorString = authorString.strip()         
            
            if len(authorString) != 0: 
                coauthorList.append(authorString)
        
            if len(authorString) != 0 and authorString not in vertexIdSet:  
                vertexIdDict[authorString] = i
                vertexIdSet.add(authorString)
                vertexIdList.append(authorString)
                i += 1 
        
        iterator = itertools.permutations(range(len(coauthorList)), 2)
        
        for j, k in iterator: 
            vId1 = coauthorList[j]
            vId2 = coauthorList[k]            
            
            if (vertexIdDict[vId1], vertexIdDict[vId2]) not in edgeSet and (vertexIdDict[vId2], vertexIdDict[vId1]) not in edgeSet: 
                edgeArray.append([vertexIdDict[vId1], vertexIdDict[vId2]])
                edgeSet.add((vertexIdDict[vId1], vertexIdDict[vId2]))

graph = igraph.Graph()
graph.add_vertices(len(vertexIdSet))
graph.add_edges(edgeArray) 

print(graph.summary())

sortedNames = sorted(vertexIdDict.keys(), key=lambda name: name.split()[-1])
#for name in sortedNames: 
#    print(name)

graphFileName = path + "coauthorsGraph" 
graph.save(graphFileName, "edgelist")

logging.debug("Saved graph as " + graphFileName)