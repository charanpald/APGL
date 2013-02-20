
"""
Experiment with the modularity representation of a matrix. 
"""
import numpy 
from apgl.graph.SparseGraph import SparseGraph 
from apgl.graph.GraphUtils import GraphUtils 

numpy.set_printoptions(suppress=True, precision=3)

numVertices = 10 
graph = SparseGraph(numVertices)

graph[0,1] = 1 
graph[0,2] = 1 
graph[0,3] = 1 
graph[1,2] = 1 
graph[1,3] = 1 
graph[2,3] = 1 
graph[1,4] = 1 
graph[4,5] = 1 
graph[4,6] = 1 
graph[5,6] = 1 
graph[6,7] = 1 
graph[7,8] = 1 
graph[7,9] = 1 
graph[8,9] = 1 

W = graph.getSparseWeightMatrix() 
B = GraphUtils.modularityMatrix(W)
u, V = numpy.linalg.eigh(B)

print(u)
print(V)