import numpy 
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from exp.influence2.MaxInfluence import MaxInfluence 
from apgl.util.PathDefaults import PathDefaults 
from exp.util.IdIndexer import IdIndexer 
import array

#Read in graph 
dataDir = PathDefaults.getDataDir() + "reputation/" 
dataFileName = dataDir + "dataset_sample.csv" 
domainFileName  = dataDir + "domains.csv" 

dataFile = open(dataFileName)
dataFile.readline() 

authorIndexer = IdIndexer("i")
articleIndexer = IdIndexer("i")
domainInds = array.array("i")
maxVals = 10000

authorIdDomainDict = {}

domainFile = open(domainFileName)
domainFile.readline() 

for line in domainFile: 
    vals = line.split("\t")
    print(vals)
    
    authorId = vals[0].strip()
    domainInd = int(vals[1])
    
    authorIdDomainDict[authorId] = domainInd


for i, line in enumerate(dataFile): 
    if i % 1000 == 0: 
        print(i)
    if i == maxVals: 
        print("Max number of iterations reached")
        break 
    
    vals = line.split("\t")
    
    authorId = vals[1].strip()
    articleId = int(vals[4]) 
    
    domainInd = authorIdDomainDict.get(authorId, -1)
    
    authorIndexer.append(authorId)
    articleIndexer.append(articleId)
    domainInds.append(domainInd)

authorInds = authorIndexer.getArray()
articleInds = articleIndexer.getArray()
domainInds = numpy.array(domainInds)
edges = numpy.c_[authorInds, articleInds]

print(numpy.max(authorInds), numpy.max(articleInds))
print((domainInds!=-1).sum())

#Coauthor graph is undirected 
graph = igraph.Graph()
graph.add_vertices(numpy.max(authorInds) + numpy.max(articleInds))
graph.add_edges(edges)

print(graph.summary())


k = 10
#rank1 = MaxInfluence.celf(graph, k, 5, p=0.5)
#print(rank1)

#scores1 = graph.eigenvector_centrality(directed=True)
scores = graph.betweenness()
rank2 = numpy.flipud(numpy.argsort(scores)) 
print(rank2)

scores = graph.pagerank()
rank3 = numpy.flipud(numpy.argsort(scores)) 
print(rank3)


print(len(graph.components()))
compSizes = [len(x) for x in graph.components()]
print(numpy.max(compSizes))