import unittest
from apgl.io.MDLGraphsReader import MDLGraphsReader
from apgl.util.PathDefaults import PathDefaults 


class  MDLGraphsReaderTestCase(unittest.TestCase):
    def testMDLGraphsReader(self):
        reader = MDLGraphsReader()
        dir = PathDefaults.getDataDir()
        fileName = dir + "test/testGraphs.mdl"

        graphs = reader.readFromFile(fileName)
        self.assertEquals(len(graphs), 2)

        #Check the first graph
        self.assertEquals(graphs[0].getNumVertices(), 26)
        self.assertEquals(graphs[0].getNumEdges(), 28)

        def getEdge(graph, i, j):
            return graph.getEdge(i-1, j-1)

        self.assertEquals(getEdge(graphs[0], 1, 6), 1)
        self.assertEquals(getEdge(graphs[0], 1,  2), 1)
        self.assertEquals(getEdge(graphs[0], 1, 18), 1)
        self.assertEquals(getEdge(graphs[0],2,  3), 1)
        self.assertEquals(getEdge(graphs[0],2, 19), 1)
        self.assertEquals(getEdge(graphs[0],3,  4), 1)
        self.assertEquals(getEdge(graphs[0],3, 20), 1)
        self.assertEquals(getEdge(graphs[0],4, 10), 1)
        self.assertEquals(getEdge(graphs[0],4,  5), 1)
        self.assertEquals(getEdge(graphs[0],5,  6), 1)
        self.assertEquals(getEdge(graphs[0],5,  7), 1)
        self.assertEquals(getEdge(graphs[0],6, 21), 1)
        self.assertEquals(getEdge(graphs[0],7,  8), 1)
        self.assertEquals(getEdge(graphs[0],7, 22), 1)
        self.assertEquals(getEdge(graphs[0],8,  9), 1)
        self.assertEquals(getEdge(graphs[0],8, 23), 1)
        self.assertEquals(getEdge(graphs[0],9, 14), 1)
        self.assertEquals(getEdge(graphs[0],9, 10), 1)
        self.assertEquals(getEdge(graphs[0],10, 11), 1)
        self.assertEquals(getEdge(graphs[0],11, 12), 1)
        self.assertEquals(getEdge(graphs[0],11, 24), 1)
        self.assertEquals(getEdge(graphs[0],12, 13), 1)
        self.assertEquals(getEdge(graphs[0],12, 25), 1)
        self.assertEquals(getEdge(graphs[0],13, 14), 1)
        self.assertEquals(getEdge(graphs[0],13, 15), 1)
        self.assertEquals(getEdge(graphs[0],14, 26), 1)
        self.assertEquals(getEdge(graphs[0],15, 16), 1)
        self.assertEquals(getEdge(graphs[0],15, 17), 1)

        #Check the second graph
        self.assertEquals(graphs[1].getNumVertices(), 19)
        self.assertEquals(graphs[1].getNumEdges(), 20)

        self.assertEquals(getEdge(graphs[1],1, 10), 1)
        self.assertEquals(getEdge(graphs[1],1,  2), 1)
        self.assertEquals(getEdge(graphs[1],1, 14), 1)
        self.assertEquals(getEdge(graphs[1],2,  3), 1)
        self.assertEquals(getEdge(graphs[1],2, 15), 1)
        self.assertEquals(getEdge(graphs[1],3,  8), 1)
        self.assertEquals(getEdge(graphs[1],3,  4), 1)
        self.assertEquals(getEdge(graphs[1],4,  5), 1)
        self.assertEquals(getEdge(graphs[1],4, 16), 1)
        self.assertEquals(getEdge(graphs[1],5,  6), 1)
        self.assertEquals(getEdge(graphs[1],5, 17), 1)
        self.assertEquals(getEdge(graphs[1],6,  7), 1)
        self.assertEquals(getEdge(graphs[1],6, 18), 1)
        self.assertEquals(getEdge(graphs[1],7,  8), 1)
        self.assertEquals(getEdge(graphs[1],8,  9), 1)
        self.assertEquals(getEdge(graphs[1],9, 10), 1)
        self.assertEquals(getEdge(graphs[1],9, 11), 1)
        self.assertEquals(getEdge(graphs[1],10, 19), 1)
        self.assertEquals(getEdge(graphs[1],11, 12), 1)
        self.assertEquals(getEdge(graphs[1],11, 13), 1)

if __name__ == '__main__':
    unittest.main()

