import numpy 
import logging
import sys
from apgl.graph import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class DictGraphProfile(object):
    def __init__(self):
        numVertices = 5000

        self.graph = DictGraph()
        
        numEdges = 1000000
        edges = numpy.zeros((numEdges, 2))
        edges[:, 0] = numpy.random.randint(0, numVertices, numEdges)
        edges[:, 1] = numpy.random.randint(0, numVertices, numEdges)
        
        self.graph.addEdges(edges)
        
        print(self.graph)
        
    def profileConnectedComponents(self):
        ProfileUtils.profile('self.graph.findConnectedComponents()', globals(), locals())
        
    def profileDepthFirstSearch(self):
        root = self.graph.getAllVertexIds()[0]        
        
        ProfileUtils.profile('self.graph.depthFirstSearch(root)', globals(), locals())
        
    def profileBreadthFirstSearch(self):
        root = self.graph.getAllVertexIds()[0]        
        
        ProfileUtils.profile('self.graph.breadthFirstSearch(root)', globals(), locals())

profiler = DictGraphProfile()
#profiler.profileDepthFirstSearch() 
#profiler.profileBreadthFirstSearch() 
profiler.profileConnectedComponents() 
