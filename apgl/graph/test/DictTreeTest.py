
from apgl.graph.DictTree import DictTree
import unittest
import numpy 

class DictGraphTest(unittest.TestCase):
    def setUp(self):
        pass

    def testInit(self):
        dictTree = DictTree()

    def testAddEdge(self):

        dictTree = DictTree()

        dictTree.addEdge("a", "b")
        dictTree.addEdge("a", "c")
        dictTree.addEdge("d", "a")

        #Add duplicate edge 
        dictTree.addEdge("a", "b")
        dictTree.addEdge("a", "c")
        dictTree.addEdge("d", "a")

        self.assertRaises(ValueError, dictTree.addEdge, "e", "a")
        
        #Add isolated edge
        self.assertRaises(ValueError, dictTree.addEdge, "r", "s")


    def testGetRoot(self):
        dictTree = DictTree()

        dictTree.addEdge("a", "b")
        dictTree.addEdge("a", "c")
        dictTree.addEdge("d", "a")

        self.assertEquals(dictTree.getRootId(), "d")

        dictTree.addEdge("e", "d")
        self.assertEquals(dictTree.getRootId(), "e")

    def testSetVertex(self):
        dictTree = DictTree()

        dictTree.setVertex("a")
        self.assertEquals(dictTree.getVertex("a"), None)
        self.assertRaises(RuntimeError, dictTree.setVertex, "b")

        dictTree.setVertex("a", 12)
        self.assertEquals(dictTree.getVertex("a"), 12)

    def testStr(self):
        dictTree = DictTree()

        dictTree.addEdge(0, 1)
        dictTree.addEdge(0, 2)
        dictTree.addEdge(2, 3)
        dictTree.addEdge(2, 4)
        dictTree.addEdge(0, 5)
        dictTree.addEdge(4, 6)


    def testDepth(self):
        dictTree = DictTree()
        self.assertEquals(dictTree.depth(), 0)
        dictTree.setVertex("a")
        self.assertEquals(dictTree.depth(), 0)

        dictTree.addEdge("a", "b")
        dictTree.addEdge("a", "c")
        dictTree.addEdge("d", "a")

        self.assertEquals(dictTree.depth(), 2)

        dictTree.addEdge("c", "e")
        self.assertEquals(dictTree.depth(), 3)

    def testCutTree(self):
        dictTree = DictTree()
        dictTree.setVertex("a", "foo")
        dictTree.addEdge("a", "b", 2)
        dictTree.addEdge("a", "c")
        dictTree.addEdge("c", "d", 5)
        dictTree.addEdge("c", "f")

        A = numpy.array([10, 2])
        dictTree.setVertex("b", A)

        newTree = dictTree.cut(2)
        self.assertEquals(newTree.getVertex("a"), "foo")
        self.assertTrue((newTree.getVertex("b") == A).all())
        self.assertEquals(newTree.getEdge("a", "b"), 2)
        self.assertEquals(newTree.getEdge("a", "c"), 1)
        self.assertEquals(newTree.getEdge("c", "d"), 5)
        self.assertEquals(newTree.getEdge("c", "f"), 1)
        self.assertEquals(newTree.getNumVertices(), dictTree.getNumVertices())
        self.assertEquals(newTree.getNumEdges(), dictTree.getNumEdges())

        newTree = dictTree.cut(1)
        self.assertEquals(newTree.getEdge("a", "b"), 2)
        self.assertEquals(newTree.getEdge("a", "c"), 1)
        self.assertEquals(newTree.getNumVertices(), 3)
        self.assertEquals(newTree.getNumEdges(), 2)

        newTree = dictTree.cut(0)
        self.assertEquals(newTree.getNumVertices(), 1)
        self.assertEquals(newTree.getNumEdges(), 0)

    def testLeaves(self):
        dictTree = DictTree()
        dictTree.setVertex("a", "foo")

        self.assertTrue(set(dictTree.leaves()) == set(["a"]))
        dictTree.addEdge("a", "b", 2)
        dictTree.addEdge("a", "c")
        dictTree.addEdge("c", "d", 5)
        dictTree.addEdge("c", "f")

        self.assertTrue(set(dictTree.leaves()) == set(["b", "d", "f"]))

        dictTree.addEdge("b", 1)
        dictTree.addEdge("b", 2)
        self.assertTrue(set(dictTree.leaves()) == set([1, 2, "d", "f"]))


    def testAddChild(self): 
        dictTree = DictTree()
        dictTree.setVertex("a", "foo")
        dictTree.addChild("a", "c", 2)
        dictTree.addChild("a", "d", 5)

        self.assertTrue(set(dictTree.leaves()) == set(["c", "d"]))
        
        self.assertEquals(dictTree.getVertex("c"), 2)
        self.assertEquals(dictTree.getVertex("d"), 5)
        
        self.assertTrue(dictTree.getEdge("a", "d"), 1.0)
        self.assertTrue(dictTree.getEdge("a", "c"), 1.0)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
