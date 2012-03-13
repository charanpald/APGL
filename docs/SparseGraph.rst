SparseGraph
===========

The SparseGraph object represents a graph with an underlying sparse adjacency matrix representation. This has 
the advantage of being efficient in memory usage for graphs with few edges. Graphs of a 1,000,000 vertices or more 
can be created with minimal memory overheads. The following is a very simple example of how to use SparseGraph:

::

	from apgl.graph import * 
	import numpy 

	numVertices = 10
	numFeatures = 3
	vList = VertexList(numVertices, numFeatures)
	vertices = numpy.random.rand(numVertices, numFeatures)
	vList.setVertices(vertices)

	graph = SparseGraph(vList)
	graph.addEdge(0, 1)
	graph.addEdge(0, 2)
	graph.addEdge(0, 3)
	graph.addEdge(2, 1)
	graph.addEdge(2, 5)
	graph.addEdge(2, 6)
	graph.addEdge(6, 9)
	
	#Note can also use the notation e.g. graph[0,1] = 1 to create an edge

	subgraph = graph.subgraph([0,1,2,3])

The code creates a new VertexList object with 10 vertices and 3 features, and fills each vertex with random numbers. The VertexList object is then used to create a SparseGraph, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that non-vector vertices 
can be added to a SparseGraph using the GeneralVertexList class in the constructor. 

You can also create a graph in which vertices can take any value using the GeneralVertexList class. Furthermore, in order to speed up certain operations on the graph, SparseGraph can be intialised with an empty sparse matrix of several types. For example, the csr_matrix allows fast out-degree computations wheareas csc_matrix is faster for finding in-degrees of directed graphs.   

::

	from apgl.graph import * 
	import numpy 
	import scipy.sparse

	numVertices = 10
	vList = GeneralVertexList(numVertices)

	weightMatrix = scipy.sparse.lil_matrix((numVertices, numVertices))
	sGraph = SparseGraph(vList, W=weightMatrix)
	sGraph.addEdge(0, 1)
	sGraph.addEdge(0, 2)
	sGraph.setVertex(0, "abc") 
	sGraph.setVertex(1, 123)
	
Here, the SparseGraph is initialised with GeneralVertexList and the first and second vertices are initialised with "abc" and 123 respectively. The sparse matrix weightMatrix passed to the constructor is used to store edge weights.  
	

Methods 
-------
.. autoclass:: apgl.graph.SparseGraph.SparseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
