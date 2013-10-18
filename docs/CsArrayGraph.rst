CsArrayGraph
===========

The CsArrayGraph object represents a graph with an underlying sparse weight matrix representation of csarray (see http://pythonhosted.org/sppy/). This has the advantage of being efficient in memory usage for graphs with few edges. Graphs of a 1,000,000 vertices or more can be created with minimal memory overheads. The following is a very simple example of how to use CsArrayGraph:

::

	from apgl.graph.CsArrayGraph import CsArrayGraph 
	import numpy 

	numVertices = 10

	graph = CsArrayGraph(numVertices)
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

The code creates a new CsArrayGraph with 10 vertices, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that numpy.array vertices can be added to a CsArrayGraph using the VertexList class in the constructor. Finally, the first and second vertices are initialised with "abc" and 123 respectively

In order to speed up certain operations on the graph, CsArrayGraph can be intialised with an empty sparse matrix of several types. For example, the csr_matrix allows fast out-degree computations wheareas csc_matrix is faster for finding in-degrees of directed graphs.   

::

	from apgl.graph.CsArrayGraph import CsArrayGraph 
	import numpy 
	import scipy.sparse

	numVertices = 10

	weightMatrix = scipy.sparse.lil_matrix((numVertices, numVertices))
	graph = CsArrayGraph(numVertices, W=weightMatrix)
	graph[0, 1] = 1 
	graph[0, 2] = 1 

	
	#Output the number of vertices
	print(graph.size)
	
Here, the CsArrayGraph is initialised with 10 vertices and the sparse matrix weightMatrix passed to the constructor is used to store edge weights.  
	

Methods 
-------
.. autoclass:: apgl.graph.CsArrayGraph.CsArrayGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
