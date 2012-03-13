
import unittest
import os
from apgl.graph.DictGraph import DictGraph
from apgl.io.SimpleGraphWriter import SimpleGraphWriter
from apgl.util.PathDefaults import PathDefaults 

class  SimpleGraphWriterTest(unittest.TestCase):
    def setUp(self):
        #Finally, try DictGraphs
        self.dctGraph1 = DictGraph(True)
        self.dctGraph1.addEdge(0, 1, 1)
        self.dctGraph1.addEdge(0, 2, 2)
        self.dctGraph1.addEdge(2, 4, 8)
        self.dctGraph1.addEdge(2, 3, 1)
        self.dctGraph1.addEdge(12, 4, 1)

        self.dctGraph2 = DictGraph(False)
        self.dctGraph2.addEdge(0, 1, 0.5)
        self.dctGraph2.addEdge(0, 2, 1)
        self.dctGraph2.addEdge(2, 4, 1)
        self.dctGraph2.addEdge(2, 3, 0.2)
        self.dctGraph2.addEdge(12, 4, 1)

    def testWriteToFile(self):
        sgw = SimpleGraphWriter()
        directory = PathDefaults.getOutputDir() + "test/"

        #Have to check the files
        fileName1 = directory + "dictTestUndirected"
        sgw.writeToFile(fileName1, self.dctGraph1)

        fileName2 = directory + "dictTestDirected"
        sgw.writeToFile(fileName2, self.dctGraph2)

        #os.remove(fileName1 + ".txt")
        #os.remove(fileName2 + ".txt")

if __name__ == '__main__':
    unittest.main()

