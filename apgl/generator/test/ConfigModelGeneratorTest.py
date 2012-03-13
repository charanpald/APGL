
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.generator.ConfigModelGenerator import ConfigModelGenerator
import unittest
import numpy 

class ConfigModelGeneratorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testInit(self):
        degSequence = numpy.array([2, 1, 2])
        generator = ConfigModelGenerator(degSequence)
        generator = ConfigModelGenerator(degSequence, degSequence)

        degSequence = numpy.array([2, 1, 3])
        generator = ConfigModelGenerator(degSequence)
        generator = ConfigModelGenerator(degSequence, degSequence)

        self.assertRaises(ValueError, ConfigModelGenerator, None)

        degSequence2 = numpy.array([2, 1, 2.2])
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence2)
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence, degSequence2)

        degSequence2 = numpy.array([2, 1, -1])
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence2)
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence, degSequence2)

        degSequence2 = numpy.array([2, 1, 4])
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence2)
        self.assertRaises(ValueError, ConfigModelGenerator, degSequence, degSequence2)

    def testGenerate(self):
        degSequence = numpy.array([2, 1, 3, 0, 0, 0, 0, 0, 0, 1])
        generator = ConfigModelGenerator(degSequence)

        numVertices = 10
        graph = SparseGraph(GeneralVertexList(numVertices))
        graph = generator.generate(graph)

        tol = 3
        self.assertTrue(numpy.linalg.norm(degSequence - graph.degreeSequence()) < tol)

        degSequence = numpy.array([2, 1, 3, 0, 2, 1, 4, 0, 0, 1])
        generator.setOutDegSequence(degSequence)
        graph.removeAllEdges()
        graph = generator.generate(graph)

        self.assertTrue(numpy.linalg.norm(degSequence - graph.degreeSequence()) < tol)

        #Test using a non-empty graph
        degSequence = numpy.array([0, 0, 0, 2, 0, 0, 0, 1, 1, 0])
        generator.setOutDegSequence(degSequence)
        oldDegSequence = graph.degreeSequence()

        self.assertRaises(ValueError, generator.generate, graph, True)
        graph = generator.generate(graph, False)

        diffSequence = graph.degreeSequence() - oldDegSequence
        self.assertTrue(numpy.linalg.norm(degSequence - diffSequence) < tol)

        #Test the case where we also have an in-degree sequence
        degSequence = numpy.array([2, 1, 3, 0, 0, 0, 0, 0, 0, 1])
        inDegSequence = numpy.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        generator = ConfigModelGenerator(degSequence, inDegSequence)

        graph = SparseGraph(GeneralVertexList(numVertices))
        self.assertRaises(ValueError, generator.generate, graph)

        graph = SparseGraph(GeneralVertexList(numVertices), False)
        graph = generator.generate(graph)

        self.assertTrue(numpy.linalg.norm(degSequence - graph.outDegreeSequence()) < tol)
        self.assertTrue(numpy.linalg.norm(inDegSequence - graph.inDegreeSequence()) < tol)

        outDegSequence = numpy.array([2, 1, 3, 0, 2, 1, 4, 0, 0, 1])
        inDegSequence = numpy.array([1, 2, 1, 1, 2, 1, 2, 1, 2, 1])
        generator.setOutDegSequence(outDegSequence)
        generator.setInDegSequence(inDegSequence)
        graph.removeAllEdges()
        graph = generator.generate(graph)

        self.assertTrue(numpy.linalg.norm(outDegSequence - graph.outDegreeSequence()) < tol)
        self.assertTrue(numpy.linalg.norm(inDegSequence - graph.inDegreeSequence()) < tol)

        #In the case that the in-degree sequence sum larger than that of the out-degree it is
        #not satisfied, but the out-degree should be. 
        inDegSequence = numpy.array([1, 2, 1, 1, 2, 1, 2, 1, 5, 6])
        generator.setInDegSequence(inDegSequence)
        graph.removeAllEdges()
        graph = generator.generate(graph)
        self.assertTrue(numpy.linalg.norm(outDegSequence - graph.outDegreeSequence()) < tol)

        #Now try the other way around
        generator.setOutDegSequence(inDegSequence)
        generator.setInDegSequence(outDegSequence)
        graph.removeAllEdges()
        graph = generator.generate(graph)
        self.assertTrue(numpy.linalg.norm(outDegSequence - graph.inDegreeSequence()) < tol)

        #Test growing graph
        outDegSequence = numpy.array([2, 1, 3, 0, 2, 1, 4, 0, 0, 1])
        inDegSequence = numpy.array([1, 2, 1, 1, 2, 1, 2, 1, 2, 1])

        generator.setOutDegSequence(outDegSequence)
        generator.setInDegSequence(inDegSequence)
        graph.removeAllEdges()
        graph = generator.generate(graph)

        newOutDegreeSequence = numpy.array([2, 1, 3, 5, 2, 1, 4, 0, 0, 1])
        newInDegreeSequence = numpy.array([2, 3, 2, 2, 3, 1, 2, 1, 2, 1])
        diffOutSequence = newOutDegreeSequence - graph.outDegreeSequence()
        diffInSequence = newInDegreeSequence - graph.inDegreeSequence()
        generator.setOutDegSequence(diffOutSequence)
        generator.setInDegSequence(diffInSequence)
        graph = generator.generate(graph, False)

        self.assertTrue(numpy.linalg.norm(newOutDegreeSequence - graph.outDegreeSequence()) < tol)
        self.assertTrue(numpy.linalg.norm(newInDegreeSequence - graph.inDegreeSequence()) < tol)

    def testGenerate2(self):
        """
        Make sure that the generated degree is less than or equal to the given degree
        
        """
        numVertices = 10

        for i in range(10): 
            degSequence = numpy.random.randint(0, 3, numVertices)
            generator = ConfigModelGenerator(degSequence)
            graph = SparseGraph(GeneralVertexList(numVertices))
            graph = generator.generate(graph)

            self.assertTrue((graph.outDegreeSequence()<=degSequence).all())

        #We try to match an evolving degree sequence 
        degSequence1 = numpy.array([0,0,1,1,1,2,2,2,3, 4])
        degSequence2 = numpy.array([2,0,3,1,2,2,2,2,3, 4])
        degSequence3 = numpy.array([2,1,4,1,2,2,2,2,3, 6])

        generator = ConfigModelGenerator(degSequence1)
        graph = SparseGraph(GeneralVertexList(numVertices))
        graph = generator.generate(graph)
        self.assertTrue((degSequence1>= graph.outDegreeSequence()).all())

        deltaSequence = degSequence2 - graph.outDegreeSequence()
        generator = ConfigModelGenerator(deltaSequence)
        graph = generator.generate(graph, False)
        self.assertTrue((degSequence2>= graph.outDegreeSequence()).all())

        deltaSequence = degSequence3 - graph.outDegreeSequence()
        generator = ConfigModelGenerator(deltaSequence)
        graph = generator.generate(graph, False)
        self.assertTrue((degSequence3>= graph.outDegreeSequence()).all())

if __name__ == '__main__':
    unittest.main()