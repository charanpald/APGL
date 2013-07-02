import numpy 
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from exp.influence2.MaxInfluence import MaxInfluence 
from apgl.util.PathDefaults import PathDefaults 
import array 

#Read in graph 
dataDir = PathDefaults.getDataDir() + "reputation/" 
dataFileName = dataDir + "dataset_sample.csv" 

dataFile = open(dataFileName)

dataFile.readline() 

authorIds = array.array("i")
articleInds = array.array("i")
maxVals = 1000

articleIndset = set([])
articleIdDict = {}
j = 0

for i, line in enumerate(dataFile): 
    print(i)
    if i == maxVals: 
        break 
    
    vals = line.split("\t")
    
    authorId = int(vals[0])
    articleId = int(vals[4])
    
    if articleId not in articleIndset: 
        articleIndset.add(articleId)
        articleIdDict[articleId] = j
        articleInd = j 
        j += 1 
    else: 
        articleInd = articleIdDict[articleId]    
    
    authorIds.append(authorId)
    articleInds.append(articleInd)

authorIds = numpy.array(authorIds)
articleInds = numpy.array(articleInds)
edges = numpy.c_[authorIds, articleInds]

graph = igraph.Graph()
graph.add_vertices(numpy.max(authorIds) + numpy.max(articleInds))
graph.add_edges(edges)

print(graph.summary())

graph.es["p"] = numpy.ones(graph.ecount())*0.5

k = 10
rank1 = MaxInfluence.celf(graph, k, 5)

#scores1 = graph.eigenvector_centrality(directed=True)
scores = graph.betweenness()
rank2 = numpy.flipud(numpy.argsort(scores)) 

scores = graph.pagerank()
rank3 = numpy.flipud(numpy.argsort(scores)) 


print(rank1)
print(rank2)
print(rank3)

print(graph.components())