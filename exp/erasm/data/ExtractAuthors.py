
"""
We will analyse the article metadata file and extract co-authors. 
"""

from apgl.util.PathDefaults import PathDefaults 
import os
import logging 
import sys 
import itertools 
import igraph 
import matplotlib.pyplot as plt 
import json 
import gc 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

path = "/local/dhanjalc/dataDump-28-11-12/" 
fileName = path + "articleMetadata-28-11-12"

if not os.path.exists(fileName): 
    path = PathDefaults.getDataDir() + "erasm/"
    fileName = path + "articleMetadata100000"

logging.debug("Loading article metadata from " + fileName)

fileObj = open(fileName, 'r')
vertexIdDict = {} 
vertexIdSet = set([])
vertexIdList = []
edgeSet = set([])
edgeArray = []

i = 0 
lineInd = 0 
emptyAuthors = 0

newEdges = []
graph = igraph.Graph()

#TODO: Write out file in real time 

for line in fileObj: 
    if lineInd % 1000 == 0: 
        print("Line " + str(lineInd) + " Author " + str(len(vertexIdSet)) + " empty author strings " + str(emptyAuthors)) 
        graph.add_edges(newEdges)
        del newEdges 
        newEdges = []
        gc.collect()
    
    articleMetaData = json.loads(line)
    
    if "authors" in articleMetaData: 
        authors = articleMetaData["authors"]
        del articleMetaData
        
        coauthorList = []
        for author in authors: 
            authorString = "".join([author["forename"], " ", author["surname"]])
            authorString = authorString.strip()         
            
            if len(authorString) != 0: 
                if authorString not in vertexIdSet: 
                    graph.add_vertex(authorString)                   
                
                coauthorList.append(authorString)
                vertexIdSet.add(authorString)
                
                del authorString 
            else: 
                emptyAuthors += 1
            
        iterator = itertools.combinations(coauthorList, 2)
        del coauthorList 
        
        for vId1, vId2 in iterator:         
            newEdges.append([vId1, vId2])

    lineInd += 1 

print(sys.getsizeof(vertexIdDict))
print(sys.getsizeof(vertexIdSet))
print(sys.getsizeof(vertexIdList))
print(sys.getsizeof(edgeSet))
print(sys.getsizeof(edgeArray))
print(sys.getsizeof(newEdges))
print(sys.getsizeof(graph))

logging.debug(graph.summary())

print(graph.neighbors("David Anderson"))

#sortedNames = sorted(vertexIdDict.keys(), key=lambda name: name.split()[-1])
#for name in sortedNames: 
#    print(name)

graphFileName = path + "coauthorsGraph" 
graph.save(graphFileName, "edgelist")

logging.debug("Saved graph as " + graphFileName)