
from apgl.graph import *
from apgl.kernel import *
from apgl.util.Util import Util
import numpy 

"""
We will test the permutation graph kernel on some simple examples and also the
random walk kernel. 
"""

#E.g. One graph with 1 edge added, 2, 3, 4 etc
# Test variation of kernel with different size graphs
# Can test subgraph matching
# Use kernel and vary value of tau

#After that, test on artificial and real data and compare with random walk kernel

numVertices = 10
numFeatures = 3

vList = VertexList(numVertices, numFeatures)
sGraph1 = SparseGraph(vList)

sGraph1.addEdge(0, 1)
sGraph1.addEdge(1, 2)
sGraph1.addEdge(0, 2)
sGraph1.addEdge(2, 3)
sGraph1.addEdge(3, 4)

sGraph2 = SparseGraph(vList)

sGraph2.addEdge(0, 1)
sGraph2.addEdge(1, 2)
sGraph2.addEdge(0, 2)
sGraph2.addEdge(2, 3)
sGraph2.addEdge(2, 4)
sGraph2.addEdge(3, 4)

tau = 1.0
linearKernel = LinearKernel()
kernel = PermutationGraphKernel(tau, linearKernel)

lmbda = 0.1
kernel2 = RandWalkGraphKernel(lmbda)

(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel.evaluate(sGraph1, sGraph2, True)
evaluation2 = kernel2.evaluate(sGraph1, sGraph2)
print(f)
print(evaluation2)
#We would expect 1 if P gets the best match, but P is orthogonal so we get a lower number 

sGraph2.addEdge(4, 5)
(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel.evaluate(sGraph1, sGraph2, True)
evaluation2 = kernel2.evaluate(sGraph1, sGraph2)
print(f)
print(evaluation2)

sGraph1.addEdge(4, 5)
(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel.evaluate(sGraph1, sGraph2, True)
evaluation2 = kernel2.evaluate(sGraph1, sGraph2)
print(f)
print(evaluation2)

#Now we test subgraph matching
#

sGraph1.removeAllEdges()
sGraph2.removeAllEdges()

#This is a subgraph of the below
sGraph1.addEdge(3, 8)
sGraph1.addEdge(7, 8)
sGraph1.addEdge(4, 7)
sGraph1.addEdge(3, 4)
sGraph1.addEdge(3, 7)

sGraph2.addEdge(0, 1)
sGraph2.addEdge(0, 2)
sGraph2.addEdge(2, 3)
sGraph2.addEdge(1, 2)
sGraph2.addEdge(3, 4)
sGraph2.addEdge(4, 6)
sGraph2.addEdge(2, 5)

#The value of f will be 3 in the exact case
(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel.evaluate(sGraph1, sGraph2, True)
evaluation2 = kernel2.evaluate(sGraph1, sGraph2)
print(f)
print(evaluation2)

#Final test - include vertex values 
perm = numpy.random.permutation(numVertices)
noise = 0.1

vLabels1 = numpy.random.rand(numVertices, numFeatures)
vLabels2 = vLabels1 + noise * numpy.random.rand(numVertices, numFeatures)
vLabels2[perm, :] = vLabels2

vList1 = VertexList(numVertices, numFeatures)
vList2 = VertexList(numVertices, numFeatures)

vList1.setVertices(vLabels1)
vList2.setVertices(vLabels2)

sGraph1 = SparseGraph(vList1)
sGraph2 = SparseGraph(vList2)

sGraph1.addEdge(3, 8)
sGraph1.addEdge(7, 8)
sGraph1.addEdge(4, 7)
sGraph1.addEdge(3, 4)
sGraph1.addEdge(3, 7)

sGraph2.addEdge(perm[3], perm[8])
sGraph2.addEdge(perm[7], perm[8])
sGraph2.addEdge(perm[4], perm[7])
sGraph2.addEdge(perm[3], perm[4])
sGraph2.addEdge(perm[3], perm[7])
#Extra edges
sGraph2.addEdge(perm[3], perm[9])
sGraph2.addEdge(perm[8], perm[9])

#TODO: Have common evaluation measure 
tau = 0.5
kernel3 = PermutationGraphKernel(tau, linearKernel)
(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel.evaluate(sGraph1, sGraph2, True)
print((kernel3.getObjectiveValue(tau, P, sGraph1, sGraph2)))

#This should give an improvement over the objective with tau!=1
(evaluation, f, P, SW1, SW2, SK1, SK2) = kernel3.evaluate(sGraph1, sGraph2, True)
print((kernel3.getObjectiveValue(tau, P, sGraph1, sGraph2)))


