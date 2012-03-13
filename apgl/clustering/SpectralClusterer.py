
import numpy
import scipy.cluster.vq as vq 
from apgl.data.Standardiser import Standardiser

"""
A spectral clustering method using the normalised Laplacian. 
"""

class SpectralClusterer(object):
    def __init__(self, k):
        """
        Initialise the object. The parameter k controls
        how many clusters to look for.

        :param k: the number of clusters
        :type k: :class:`int`
        """
        self.k = k
        self.numIterKmeans = 20 

    def cluster(self, graph):
        """
        Take a graph and cluster using the method in "On spectral clusering: analysis
        and algorithm" by Ng et al., 2001. 

        :param graph: the graph to cluster
        :type graph: :class:`apgl.graph.AbstractMatrixGraph`

        :returns:  An array of size graph.getNumVertices() of cluster membership 
        """
        L = graph.normalisedLaplacianSym()

        omega, Q = numpy.linalg.eig(L)
        inds = numpy.argsort(omega)

        #First normalise rows, then columns
        standardiser = Standardiser()
        V = standardiser.normaliseArray(Q[:, inds[0:self.k]].T).T
        V = vq.whiten(V)
        #Using kmeans2 here seems to result in a high variance
        #in the quality of clustering. Therefore stick to kmeans
        centroids, clusters = vq.kmeans(V, self.k, iter=self.numIterKmeans)
        clusters, distortion = vq.vq(V, centroids)

        return clusters
