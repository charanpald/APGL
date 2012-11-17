SparseGraph
===========

The SparseGraph object represents a graph with an underlying sparse weight matrix representation from scipy.sparse. This has the advantage of being efficient in memory usage for graphs with few edges. Graphs of a 1,000,000 vertices or more can be created with minimal memory overheads. The following is a very simple example of how to use SparseGraph:

::

	from apgl.graph import SparseGraph 
	import numpy 

	numVertices = 10

	graph = SparseGraph(numVertices)
	graph.addEdge(0, 1)
	#Note can also use the notation e.g. graph[0,1] = 1 to create an edge
	graph[0, 2] = 1 
	graph[0, 3] = 1 
	graph[2, 1] = 1 
	graph[2, 5] = 1 
	graph[2, 6] = 1 
	graph[6, 9] = 1 
	
	subgraph = graph.subgraph([0,1,2,3])
	
	graph.vlist[0] = "abc" 
	graph.vlist[1] =  123

The code creates a new SparseGraph with 10 vertices, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that numpy.array vertices can be added to a SparseGraph using the VertexList class in the constructor. Finally, the first and second vertices are initialised with "abc" and 123 respectively

In order to speed up certain operations on the graph, SparseGraph can be intialised with an empty sparse matrix of several types. For example, the csr_matrix allows fast out-degree computations wheareas csc_matrix is faster for finding in-degrees of directed graphs.   

::

	from apgl.graph import SparseGraph 
	import numpy 
	import scipy.sparse

	numVertices = 10

	weightMatrix = scipy.sparse.lil_matrix((numVertices, numVertices))
	graph = SparseGraph(numVertices, W=weightMatrix)
	graph[0, 1] = 1 
	graph[0, 2] = 1 

	
	#Output the number of vertices
	print(graph.size)
	
Here, the SparseGraph is initialised with 10 vertices and the sparse matrix weightMatrix passed to the constructor is used to store edge weights.  
	

Methods 
-------
.. autoclass:: apgl.graph.SparseGraph.SparseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
