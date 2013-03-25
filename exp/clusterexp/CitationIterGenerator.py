import datetime
import numpy
import logging 
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.DictGraph import DictGraph
from apgl.graph import SparseGraph, VertexList 
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator
from apgl.util.SparseUtils import SparseUtils 

class CitationIterGenerator(object):
    """
    A class to load the high energy physics data and generate an iterator. The 
    dataset is found in http://snap.stanford.edu/data/cit-HepTh.html
    """
    def __init__(self, minGraphSize=500, maxGraphSize=None, dayStep=30):
        
        dataDir = PathDefaults.getDataDir() + "cluster/"
        edgesFilename = dataDir + "Cit-HepTh.txt"
        dateFilename = dataDir + "Cit-HepTh-dates.txt"

        #Note the IDs are integers but can start with zero so we prefix "1" to each ID 
        edges = []
        file = open(edgesFilename, 'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()

        for line in file:
            (vertex1, sep, vertex2) = line.partition("\t")
            vertex1 = vertex1.strip()
            vertex2 = vertex2.strip()
            edges.append([vertex1, vertex2])
            
            #if vertex1 == vertex2: 
            #    print(vertex1)

        file.close()

        logging.info("Loaded edge file " + str(edgesFilename) + " with " + str(len(edges)) + " edges")

        #Keep an edge graph 
        graph = DictGraph(False)
        graph.addEdges(edges)
        logging.info("Created directed citation graph with " + str(graph.getNumEdges()) + " edges and " + str(graph.getNumVertices()) + " vertices")

        #Read in the dates articles appear in a dict which used the year and month
        #as the key and the value is a list of vertex ids. For each month we include
        #all papers uploaded that month and those directed cited by those uploads. 
        startDate = datetime.date(1990, 1, 1)

        file = open(dateFilename, 'r')
        file.readline()
        numLines = 0 
        subgraphIds = []

        for line in file:
            (id, sep, date) = line.partition("\t")
            id = id.strip()
            date = date.strip()
            

            inputDate = datetime.datetime.strptime(date.strip(), "%Y-%m-%d")
            inputDate = inputDate.date()

            if graph.vertexExists(id):
                tDelta = inputDate - startDate
                            
                graph.vertices[id] = tDelta.days 
                subgraphIds.append(id)
                
                #If a paper cites another, it must have been written before 
                #the citing paper - enforce this rule. 
                for neighbour in graph.neighbours(id): 
                    if graph.getVertex(neighbour) == None: 
                        graph.setVertex(neighbour, tDelta.days) 
                        subgraphIds.append(neighbour)
                    elif tDelta.days < graph.getVertex(neighbour): 
                        graph.setVertex(neighbour, tDelta.days) 
                        
            numLines += 1 
            
        file.close()
        
        subgraphIds = set(subgraphIds)
        graph = graph.subgraph(list(subgraphIds))
        logging.debug(graph)
        logging.info("Loaded date file " + str(dateFilename) + " with " + str(len(subgraphIds)) + " dates and " + str(numLines) + " lines")

        W = graph.getSparseWeightMatrix()
        W = W + W.T
        
        vList = VertexList(W.shape[0], 1)
        vList.setVertices(numpy.array([graph.getVertices(graph.getAllVertexIds())]).T)
        
        #Note: we have 16 self edges and some two-way citations so this graph has fewer edges than the directed one 
        self.graph = SparseGraph(vList, W=W)
        logging.debug(self.graph)
        
        #Now pick the max component 
        components = self.graph.findConnectedComponents()
        self.graph = self.graph.subgraph(components[0])
        
        logging.debug("Largest component graph: " + str(self.graph))
        
        self.minGraphSize = minGraphSize
        self.maxGraphSize = maxGraphSize 
        self.dayStep = dayStep 
        
    def getIterator(self):
        """
        Return an iterator which outputs the citation graph for each month. Note
        that the graphs are undirected but we make them directed.
        """
        vertexArray = self.graph.getVertexList().getVertices()
        dates = vertexArray[:, 0]
        firstVertex = numpy.argmin(dates)
        
        dayList = range(int(numpy.min(dates)), int(numpy.max(dates)), self.dayStep)
        dayList.append(numpy.max(dates))
        
        subgraphIndicesList = []

        #Generate subgraph indices list 
        for i in dayList:
            subgraphIndices = numpy.nonzero(dates <= i)[0]
            
            #Check subgraphIndices are sorted 
            subgraphIndices = numpy.sort(subgraphIndices)
            currentSubgraph = self.graph.subgraph(subgraphIndices)
            compIndices = currentSubgraph.depthFirstSearch(list(subgraphIndices).index(firstVertex))
            subgraphIndices =  subgraphIndices[compIndices]        
            
            if self.maxGraphSize != None and subgraphIndices.shape[0] > self.maxGraphSize: 
                break 
            
            print(subgraphIndices.shape[0])
            
            if subgraphIndices.shape[0] >= self.minGraphSize: 
                subgraphIndicesList.append(subgraphIndices)
                
        iterator = IncreasingSubgraphListIterator(self.graph, subgraphIndicesList)
        return iterator 
