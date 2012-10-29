"""
Try to replicate the toy dateset in the paper by Fowlkes et al., "Spectral Grouping 
Using the Nystrom Method" and compute the spectrum. 
"""


import numpy 
import scipy.sparse 
import matplotlib.pyplot as plt 
from apgl.graph import SparseGraph, GeneralVertexList, GraphUtils  
from exp.sandbox.Nystrom import Nystrom 

numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

numVertices = 150 
X = numpy.zeros((numVertices, 2))

#Create circle 
radius = 5 
noise = 0.3
angles = numpy.random.rand(100)*2*numpy.pi
X[0:100, 0] = radius*numpy.sin(angles)
X[0:100, 1] = radius*numpy.cos(angles)

X[0:100, :] += numpy.random.randn(100, 2)*noise 

#Create blob 
R = 1 
centre = numpy.array([5-R, 0])
X[100:, :] = centre + numpy.random.randn(50, 2)*noise 

plt.figure(0)
plt.scatter(X[0:100, 0], X[0:100, 1], c="r")
plt.scatter(X[100:, 0], X[100:, 1], c="b")


#Compute weight matrix 
sigma = 0.2 
W = numpy.zeros((numVertices, numVertices))

for i in range(numVertices): 
    for j in range(numVertices): 
        W[i, j] = numpy.exp(-(numpy.linalg.norm(X[i, :] - X[j, :])**2)/(2*sigma**2))


graph = SparseGraph(GeneralVertexList(numVertices))
graph.setWeightMatrix(W)
L = graph.normalisedLaplacianSym()

#L = GraphUtils.shiftLaplacian(scipy.sparse.csr_matrix(W)).todense()
n = 100 
omega, Q = numpy.linalg.eigh(L)
omega2, Q2 = Nystrom.eigpsd(L, n)

print(omega)
print(omega2)

plt.figure(1)
plt.plot(numpy.arange(omega.shape[0]), omega)
plt.plot(numpy.arange(omega2.shape[0]), omega2)
plt.show()