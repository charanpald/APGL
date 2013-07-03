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

authorInds = array.array("i")
articleInds = array.array("i")
maxVals = 10000

authorIdSet = set([])
authorIdDict = {}
p = 0

articleIdSet = set([])
articleIdDict = {}
j = 0

for i, line in enumerate(dataFile): 
    print(i)
    if i == maxVals: 
        break 
    
    vals = line.split("\t")
    
    authorId = int(vals[0])
    articleId = int(vals[4])
   
    if authorId not in authorIdSet: 
        authorIdSet.add(authorId)
        authorIdDict[authorId] = p
        authorInd = p 
        p += 1 
    else: 
        authorInd = authorIdDict[authorId]       
   
    if articleId not in articleIdSet: 
        articleIdSet.add(articleId)
        articleIdDict[articleId] = j
        articleInd = j 
        j += 1 
    else: 
        articleInd = articleIdDict[articleId]    
    
    authorInds.append(authorInd)
    articleInds.append(articleInd)

authorInds = numpy.array(authorInds)
articleInds = numpy.array(articleInds)
edges = numpy.c_[authorInds, articleInds]

graph = igraph.Graph()
graph.add_vertices(numpy.max(authorInds) + numpy.max(articleInds))
graph.add_edges(edges)

print(graph.summary())

graph.es["p"] = numpy.ones(graph.ecount())*0.5

k = 10
#rank1 = MaxInfluence.celf(graph, k, 5)
#print(rank1)

#scores1 = graph.eigenvector_centrality(directed=True)
scores = graph.betweenness()
rank2 = numpy.flipud(numpy.argsort(scores)) 
print(rank2)

scores = graph.pagerank()
rank3 = numpy.flipud(numpy.argsort(scores)) 
print(rank3)


print(len(graph.components()))