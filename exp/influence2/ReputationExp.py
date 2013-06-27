import numpy 
import ctypes
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from exp.influence2.MaxInfluence import MaxInfluence 


#Generate random graph and then rankings 
n = 100 
m = 5 

k = 100 

graph = igraph.Graph.Barabasi(n, m, directed=True)
graph.es["p"] = numpy.random.rand(graph.ecount())
print(graph.summary())


rank1 = MaxInfluence.greedyMethod(graph, k, 5)

#scores1 = graph.eigenvector_centrality(directed=True)
scores = graph.betweenness()
rank2 = numpy.flipud(numpy.argsort(scores)) 

scores = graph.pagerank()
rank3 = numpy.flipud(numpy.argsort(scores)) 


print(rank1)
print(rank2)
print(rank3)