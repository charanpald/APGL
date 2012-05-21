import unittest
import numpy
import logging
from exp.sandbox.predictors.GrowthLearner import GrowthLearner
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
 

class  GrowthLearnerTest(unittest.TestCase):
    def setUp(self):
        pass

    def testLearnModel(self):
        numVertices = 100
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        p = 0.2
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        vertexIndices = list(range(0, numVertices))

        k = 2
        learner = GrowthLearner(k)

        tol = 10**-1

        #Lets test the values of alpha on a series of Erdos-Renyi graphs 
        for i in range(1, 6):
            p = float(i)/10
            graph.removeAllEdges()
            graph = generator.generate(graph)

            alpha = learner.learnModel(graph, vertexIndices)
            logging.debug((numpy.linalg.norm(alpha - numpy.array([p, 0]))))
            #self.assertTrue(numpy.linalg.norm(alpha - numpy.array([p, 0])) < tol)


        #Now test the learning on some preferencial attachment graphs
        ell = 10
        m = 8

        vertexIndices = list(range(ell, numVertices))
        graph.removeAllEdges()
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)

        alpha = learner.learnModel(graph, vertexIndices)
        logging.debug(alpha)


if __name__ == '__main__':
    unittest.main()

