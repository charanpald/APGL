
"""
We will analyse the article metadata file and extract co-authors. 
"""

from apgl.util.PathDefaults import PathDefaults 
import os
import logging 
import sys 
import itertools 
import json 
from apgl.util.ProfileUtils import ProfileUtils 
import numpy
import scipy 
import scipy.io 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def saveAuthors(): 
    
    path = "/local/dhanjalc/dataDump-28-11-12/" 
    fileName = path + "articleMetadata500000"    
    
    if not os.path.exists(fileName): 
        path = PathDefaults.getDataDir() + "erasm/"

    
    fileName = path + "articleMetadata1000000" 
    
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
    
    edgeFileName = PathDefaults.getOutputDir() + "edges.txt"
    edgesFile = open(edgeFileName, "w")
    lineBuffer = ""
    
    for line in fileObj:    
        if lineInd % 1000 == 0: 
            print("Line " + str(lineInd) + " Author " + str(len(vertexIdSet)) + " empty author strings " + str(emptyAuthors)) 
            if len(lineBuffer) != 0:
                edgesFile.write(lineBuffer)
            lineBuffer = ""
        
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
                        vertexIdDict[authorString] = len(vertexIdSet)
                        vertexIdSet.add(authorString)
                    
                    coauthorList.append(authorString)
                                    
                    del authorString 
                else: 
                    emptyAuthors += 1
                
            iterator = itertools.combinations(coauthorList, 2)
            del coauthorList 
            
            for vId1, vId2 in iterator:         
                #Note that we will have duplicate edges 
                lineBuffer += str(vertexIdDict[vId1]) + ", " + str(vertexIdDict[vId2]) + "\n"
    
        lineInd += 1 
    
    edgesFile.close()
    
    print(sys.getsizeof(vertexIdDict))
    print(sys.getsizeof(vertexIdSet))
    print(sys.getsizeof(vertexIdList))
    print(sys.getsizeof(edgeSet))
    print(sys.getsizeof(edgeArray))
    
    logging.debug("Saved edges as " + edgeFileName)
    
def saveRatingMatrix(): 
    """
    Take the coauthor graph above and make vertices indexed from 0 then save 
    as matrix market format. 
    """    
    edgeFileName = PathDefaults.getOutputDir() + "erasm/edges2.txt"
    
    logging.debug("Reading edge list")
    edges = numpy.loadtxt(edgeFileName, delimiter=",", dtype=numpy.int)
    logging.debug("Total number of edges: " + str(edges.shape[0]))
    
    vertexIdDict = {} 
    vertexIdSet = set([])
    
    i = 0 
        
    for edge in edges:
        if edge[0] not in vertexIdSet: 
            vertexIdDict[edge[0]] = i
            vertexIdSet.add(edge[0])
            i += 1 
         
        if edge[1] not in vertexIdSet: 
            vertexIdDict[edge[1]] = i 
            vertexIdSet.add(edge[1])
            i += 1 

    n = len(vertexIdDict)    
    R = scipy.sparse.lil_matrix((n, n))
    logging.debug("Creating sparse matrix")
    
    for edge in edges:
        R[vertexIdDict[edge[0]], vertexIdDict[edge[1]]] += 1 
        R[vertexIdDict[edge[1]], vertexIdDict[edge[0]]] += 1 
        
    logging.debug("Created matrix " + str(R.shape) + " with " + str(R.getnnz()) + " non zeros")    

    R = R.tocsr()    
    
    minCoauthors = 20
    logging.debug("Removing vertices with <" + str(minCoauthors) + " coauthors")
    nonzeros = R.nonzero()    
    inds = numpy.arange(nonzeros[0].shape[0])[numpy.bincount(nonzeros[0]) >= minCoauthors]
    R = R[inds, :][:, inds]
    logging.debug("Matrix has shape " + str(R.shape) + " with " + str(R.getnnz()) + " non zeros")    
        
    matrixFileName = PathDefaults.getOutputDir() + "erasm/R"
    scipy.io.mmwrite(matrixFileName, R)
    logging.debug("Wrote matrix to file " + matrixFileName)
    
#saveAuthors()
saveRatingMatrix()