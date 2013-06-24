
import ctypes; 
ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
import igraph 


#Generate random graph and then rankings 
n = 1000 
m = 5 


graph = igraph.Graph.Barabasi(n, m, directed=True)
print(graph.summary())


scores1 = graph.eigenvector_centrality(directed=True)
scores2 = graph.betweenness()
scores3 = graph.pagerank()

print(scores1)