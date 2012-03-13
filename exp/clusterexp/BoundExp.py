import numpy
import matplotlib.pyplot as plt
from apgl.data.Standardiser import Standardiser
from apgl.data.FeatureGenerator import FeatureGenerator
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from sklearn.cluster import KMeans
"""
Test the error bound for the clustering process. 
"""
k = 3
numVertices = 15

graph1 = SparseGraph(GeneralVertexList(numVertices))
cluster1 = numpy.array([[0,1], [0,2], [1,3], [2,3], [3,4], [4,5]])
graph1.addEdges(cluster1)
cluster2 = numpy.array([[5,7], [5,8], [7,9], [8,9], [6,7]])
graph1.addEdges(cluster2)
cluster3 = numpy.array([[6,10], [10,11], [10,12], [11,12], [11,13], [12,14]])
graph1.addEdges(cluster3)

graph2 = SparseGraph(GeneralVertexList(numVertices))
cluster1 = numpy.array([[0,1], [0,2], [1,3], [2,3], [3,4], [0,3],[4,5]])
graph2.addEdges(cluster1)
cluster2 = numpy.array([[5,7], [5,8], [7,9], [8,9], [6,7]])
graph2.addEdges(cluster2)
cluster3 = numpy.array([[6,10], [10,11], [10,12], [11,12], [11,13], [12,14]])
graph2.addEdges(cluster3)

L1 = graph1.normalisedLaplacianSym()
L2 = graph2.normalisedLaplacianSym()

l1, U = numpy.linalg.eig(L1)
inds = numpy.argsort(l1)[0:k]
U = U[:, inds]
U = Standardiser().normaliseArray(U.T).T

l2, V = numpy.linalg.eig(L2)
inds = numpy.argsort(l2)[0:k]
V = V[:, inds]
V = Standardiser().normaliseArray(V.T).T

kmeans = KMeans(k)
kmeans.fit(U)
C = FeatureGenerator().categoricalToIndicator(numpy.array([kmeans.labels_]).T, numpy.array([0])) 

kmeans.fit(V)
D = FeatureGenerator().categoricalToIndicator(numpy.array([kmeans.labels_]).T, numpy.array([0]))

#We know U and C also a pertubation delta
delta = (numpy.linalg.norm(U)**2)/2

CTilde = C.dot(numpy.diag(numpy.sum(C, 0)**-1))
DTilde = D.dot(numpy.diag(numpy.sum(D, 0)**-1))

M = CTilde.T.dot(U)
N = DTilde.T.dot(V)

#print(numpy.linalg.norm(U - V)**2)

#Want to construct a Z matrix
Z = numpy.zeros((M.shape[0], M.shape[0]))

for i in range(M.shape[0]):
    for j in range(M.shape[0]):
        if i != j:
            Z[i, j] = numpy.linalg.norm(M[i, :] - N[j, :])**2

print(Z)
print(numpy.sum(Z))

#Now compute ||Z||_1 indirectly


Z2 = 0
for i in range(C.shape[1]):
    Z2 += 1/(C[:, i].sum()) + 1/(D[:, i].sum())
print(Z2)
Z2 *= (k-1)
Z2 += 2*numpy.trace(CTilde.T.dot(U).dot(V.T).dot(DTilde))
Z2 += -2*numpy.sum(CTilde.T.dot(U).dot(V.T).dot(DTilde))
print(Z2)

print(numpy.trace(CTilde.T.dot(U).dot(U.T).dot(CTilde)))

Z3 = (k-1)*numpy.trace(CTilde.T.dot(U).dot(U.T).dot(CTilde))
Z3 += -2*numpy.sum(CTilde.T.dot(U).dot(V.T).dot(DTilde))
Z3 += (k-1)*numpy.trace(DTilde.T.dot(V).dot(V.T).dot(DTilde))
Z3 += 2*numpy.trace(CTilde.T.dot(U).dot(V.T).dot(DTilde))

print(Z3)

#plt.scatter(U[0:numExamplesPerCluster, 0], U[0:numExamplesPerCluster, 1], c='b')
#plt.scatter(U[numExamplesPerCluster:, 0], U[numExamplesPerCluster:, 1], c='r')
#plt.show()