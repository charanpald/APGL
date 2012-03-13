
"""
Test eigenvalue solvers for sparse and dense matrices 
"""
import time 
import numpy
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
from pyamg import smoothed_aggregation_solver
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList

numpy.set_printoptions(suppress=True, precision=4, linewidth=100)
numpy.random.seed(21)

p = 0.001
numVertices = 10000
generator = ErdosRenyiGenerator(p)
graph = SparseGraph(GeneralVertexList(numVertices))
graph = generator.generate(graph)

print("Num vertices = " + str(graph.getNumVertices()))
print("Num edges = " + str(graph.getNumEdges()))

L = graph.normalisedLaplacianSym(sparse=True)
#L = csr_matrix(L)
print("Created Laplacian")

#ml = smoothed_aggregation_solver(L)
#M = ml.aspreconditioner()


start = time.clock()
w,V = scipy.sparse.linalg.eigsh(L, k=20)
totalTime = time.clock() - start
print(totalTime)

#print(w)

#Eigsh is quite fast 