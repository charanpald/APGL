"""
A class to take a sequence of graphs in which one vertex is added at a time in
successive graphs and learn how the growth is implemented.
"""
import numpy

class GrowthLearner(object):
    def __init__(self, k):
        self.k = k 

    def learnModel(self, graph, vertexIndices):
        """
        Learn a model with the given graph and assume vertices are added in the
        order given in vertexIndices. 
        """
        A = graph.getWeightMatrix()

        BB = numpy.zeros((self.k, self.k))
        Bq = numpy.zeros((self.k, 1))

        for i in range(1, len(vertexIndices)-1):
            Ai = A[numpy.ix_(vertexIndices[0:i], vertexIndices[0:i])]

            Bi = numpy.zeros((i+1, self.k))
            Bi[:, 0] = numpy.ones(i+1)

            for j in range(1, self.k):
                Bi[0:i, j] = numpy.sum(Ai, 1)
                if j != self.k-1:
                    Ai = numpy.dot(Ai, Ai)

            normVector = numpy.sqrt(sum(Bi**2, 0))
            Bi = Bi / (normVector + (normVector == 0))
            
            qip1 = numpy.array([A[vertexIndices[0:i+1], vertexIndices[i+1]]]).T
            
            BB = BB + numpy.dot(Bi.T, Bi)
            Bq = Bq + numpy.dot(Bi.T, qip1)

        self.alpha = numpy.dot(numpy.linalg.inv(BB), Bq).ravel()

        return self.alpha 




        
        

        