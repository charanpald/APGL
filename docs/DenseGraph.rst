DenseGraph
===========

A graph who edges are represented by a dense (non-sparse) weight matrix, and is otherwise similar to SparseGraph. The following is a very simple example of how to use DenseGraph:

::

	from apgl.graph import * 
	import numpy 
	
	numVertices = 10
	numFeatures = 3
	vList = VertexList(numVertices, numFeatures)
	vertices = numpy.random.rand(numVertices, numFeatures)
	vList.setVertices(vertices)

	graph = DenseGraph(vList)
	graph.addEdge(0, 1)
	graph.addEdge(0, 2)
	graph.addEdge(0, 3)
	graph.addEdge(2, 1)
	graph.addEdge(2, 5)
	graph.addEdge(2, 6)
	graph.addEdge(6, 9)
	
	#Note can also use the notation e.g. graph[0,1] = 1 to create an edge

	subgraph = graph.subgraph([0,1,2,3])	
	
The code creates a new VertexList object with 10 vertices and 3 features, and fills each vertex with random numbers. The VertexList object is then used to create a DenseGraph, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that non-vector vertices 
can be added to a DenseGraph using the GeneralVertexList class in the constructor. 

Methods 
-------
.. autoclass:: apgl.graph.DenseGraph.DenseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
