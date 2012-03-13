PySparseGraph
=============

The PySparseGraph object represents a graph with an underlying sparse adjacency matrix representation implemented using Pysparse. Therefore you must install Pysparse (http://pysparse.sourceforge.net/) in order to use this class. PySparseGraph has the advantage of being efficient in memory usage for graphs with few edges, and also relatively fast as Pysparse is implemented using C. Graphs of a 1,000,000 vertices or more can be created with minimal memory overheads. The following is a very simple example of how to use PySparseGraph:

::

	from apgl.graph import * 
	import numpy 

	numVertices = 10
	numFeatures = 3
	vList = VertexList(numVertices, numFeatures)
	vertices = numpy.random.rand(numVertices, numFeatures)
	vList.setVertices(vertices)

	graph = PySparseGraph(vList)
	graph.addEdge(0, 1)
	graph.addEdge(0, 2)
	graph.addEdge(0, 3)
	graph.addEdge(2, 1)
	graph.addEdge(2, 5)
	graph.addEdge(2, 6)
	graph.addEdge(6, 9)
	
	#Note can also use the notation e.g. graph[0,1] = 1 to create an edge

	subgraph = graph.subgraph([0,1,2,3])

The code creates a new VertexList object with 10 vertices and 3 features, and fills each vertex with random numbers. The VertexList object is then used to create a PySparseGraph, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that non-vector vertices can be added to a PySparseGraph using the GeneralVertexList class in the constructor. 
	

Methods 
-------
.. autoclass:: apgl.graph.PySparseGraph.PySparseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
