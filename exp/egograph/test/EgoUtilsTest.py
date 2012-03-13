'''
Created on 20 Aug 2009

@author: charanpal
'''
import unittest
import numpy 
from apgl.egograph.EgoUtils import EgoUtils
from apgl.data.ExamplesList import ExamplesList
from apgl.util.PathDefaults import PathDefaults


class EgoUtilsTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testGraphFromMatFile(self):
        matFileName = PathDefaults.getDataDir() +  "infoDiffusion/EgoAlterTransmissions1000.mat"
        sGraph = EgoUtils.graphFromMatFile(matFileName)
        
        examplesList = ExamplesList.readFromMatFile(matFileName)
        numFeatures = examplesList.getDataFieldSize("X", 1)
        
        self.assertEquals(examplesList.getNumExamples(), sGraph.getNumEdges())
        self.assertEquals(examplesList.getNumExamples()*2, sGraph.getNumVertices())
        self.assertEquals(numFeatures/2+1, sGraph.getVertexList().getNumFeatures())
        
        #Every even vertex has information, odd does not 
        for i in range(0, sGraph.getNumVertices()): 
            vertex = sGraph.getVertex(i)
            
            if i%2 == 0: 
                self.assertEquals(vertex[sGraph.getVertexList().getNumFeatures()-1], 1)
            else: 
                self.assertEquals(vertex[sGraph.getVertexList().getNumFeatures()-1], 0)
                
        #Test the first few vertices are the same 
        for i in range(0, 10): 
            vertex1 = sGraph.getVertex(i*2)[0:numFeatures/2]
            vertex2 = sGraph.getVertex(i*2+1)[0:numFeatures/2]
            vertexEx1 = examplesList.getSubDataField("X", numpy.array([i])).ravel()[0:numFeatures/2]
            vertexEx2 = examplesList.getSubDataField("X", numpy.array([i])).ravel()[numFeatures/2:numFeatures]
            
            self.assertTrue((vertex1 == vertexEx1).all())
            self.assertTrue((vertex2 == vertexEx2).all())

    def testAverageHopDistance(self):
        transmissions = [numpy.array([[0, 1], [3, 4]])]

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 1)

        transmissions = []
        transmissions.append(numpy.array([[0, 1], [3, 4]]))
        transmissions.append(numpy.array([[4, 5]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 1.5)

        transmissions.append(numpy.array([[5, 6]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 2)

        transmissions.append(numpy.array([[2, 8]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 5.0/3)

        #Try an example with two people sending to one node
        transmissions = []
        transmissions.append(numpy.array([[0, 1], [2, 1]]))
        transmissions.append(numpy.array([[1, 5]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 1.5)

        #Try a straight line
        transmissions = []
        transmissions.append(numpy.array([[0, 1]]))
        transmissions.append(numpy.array([[1, 2]]))
        transmissions.append(numpy.array([[2, 3]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 3)

        #A branch
        transmissions.append(numpy.array([[2, 4]]))

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 4)

        #Try empty transmissions
        transmissions = []

        self.assertEquals(EgoUtils.averageHopDistance(transmissions), 0)

    def testReceiversPerSender(self):
        #Try empty transmissions
        transmissions = []
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 0)

        transmissions = [numpy.array([[0, 1], [3, 4]])]
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 1)

        transmissions = [numpy.array([[0, 1], [0, 2]])]
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 2)

        transmissions = [numpy.array([[0, 1], [0, 2], [1, 3]])]
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 1.5)

        transmissions = [numpy.array([[0, 1], [0, 2], [1, 3]]), numpy.array([[0, 5]])]
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 2)

        transmissions = [numpy.array([[0, 2], [1, 2]])]
        self.assertEquals(EgoUtils.receiversPerSender(transmissions), 0.5)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()