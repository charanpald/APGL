"""
A set of benchmark tests in order to compare the speed of algorithms under different 
graph types. 
"""
import numpy 
import logging 
import time 
from apgl.graph import DenseGraph, SparseGraph, PySparseGraph, DictGraph 

numpy.set_printoptions(suppress=True, precision=3)

class GraphIterator:
    def __init__(self, numVertices, sparseOnly=False, numEdges=0): 
        self.numVertices = numVertices 
        self.sparseOnly = sparseOnly 
        
        self.graphList = [] 
        
        if not self.sparseOnly: 
            self.graphList.append(DenseGraph(numVertices))

        self.graphList.append(SparseGraph(numVertices, frmt="lil"))
        self.graphList.append(SparseGraph(numVertices, frmt="csc"))
        self.graphList.append(SparseGraph(numVertices, frmt="csr"))
        self.graphList.append(PySparseGraph(numVertices))
        self.graphList.append(DictGraph())

        self.i = 0 

    def __iter__(self): 
        return self 
        
    def next(self): 
        if self.i==len(self.graphList): 
            raise StopIteration 
        else: 
            graph = self.graphList[self.i]
            self.i += 1 
            return graph 
            
    def getNumGraphs(self): 
        return len(self.graphList)

def generateEdges(): 
    edgeList = [] 
    
    density = 0.01 
    numVertices = numpy.array([200, 500, 1000]) 
    
    for i in numVertices:  
        numEdges = (i**2) * density
        
        
        edges = numpy.zeros((numEdges, 2))
        edges[:, 0] = numpy.random.randint(0, i, numEdges)
        edges[:, 1] = numpy.random.randint(0, i, numEdges)
        
        edgeList.append((i, edges))
        
    return edgeList 
            
    
def benchmark(edgeList): 
    iterator = GraphIterator(100)
    numGraphTypes = iterator.getNumGraphs() 
    addEdgesTimeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    neighbourTimeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    depthTimeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    breadthTimeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    componentsTimeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    i = 0 
    
    for numVertices, edges in edgeList: 
        print("Timing graphs of size " + str(numVertices) + " with " + str(edges.shape[0]) + " edges")
        
        iterator = GraphIterator(numVertices)
        j = 0 
                
        for graph in iterator:
            print("Add edges benchmark on " + str(graph))    
            startTime = time.clock()
            graph.addEdges(edges)      
            addEdgesTimeArray[i, j] =  time.clock() - startTime
                       
            vertexIds = graph.getAllVertexIds()            
            
            print("Neighbours benchmark on " + str(graph))    
            startTime = time.clock()
            for k in range(100): 
                graph.neighbours(vertexIds[k])      
            neighbourTimeArray[i, j] =  time.clock() - startTime
            
            print("Depth first search benchmark on " + str(graph))    
            startTime = time.clock()
            for k in range(5): 
                graph.depthFirstSearch(vertexIds[k])      
            depthTimeArray[i, j] =  time.clock() - startTime     
            
            print("Breadth first search benchmark on " + str(graph))    
            startTime = time.clock()
            for k in range(5): 
                graph.breadthFirstSearch(vertexIds[k])      
            breadthTimeArray[i, j] =  time.clock() - startTime     
            
            print("Find components benchmark on " + str(graph))    
            startTime = time.clock()
            graph.findConnectedComponents()      
            componentsTimeArray[i, j] =  time.clock() - startTime               
            
            j += 1
            
        i +=1 
            
    return addEdgesTimeArray, neighbourTimeArray, depthTimeArray, breadthTimeArray, componentsTimeArray 
    
edgeList = generateEdges() 
#timeArray = benchmarkAddEdges(edgeList)
addEdgesTimeArray, neighbourTimeArray, depthTimeArray, breadthTimeArray, componentsTimeArray  = benchmark(edgeList)

print(addEdgesTimeArray)
print(neighbourTimeArray)
print(depthTimeArray)
print(breadthTimeArray)
print(componentsTimeArray)
