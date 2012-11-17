DenseGraph
===========

A graph who edges are represented by a dense (numpy.ndarray) weight matrix, and is otherwise similar to SparseGraph. The following is a very simple example of how to use DenseGraph:

::

	from apgl.graph import DenseGraph 
	import numpy 
	
	numVertices = 10

	graph = DenseGraph(numVertices)
	graph[0, 2] = 1 
	graph[0, 3] = 1 
	graph[2, 1] = 1 
	graph[2, 5] = 1 
	graph[2, 6] = 1 
	graph[6, 9] = 1 
	
	subgraph = graph.subgraph([0,1,2,3])
	
	graph.vlist[0] = "abc" 
	graph.vlist[1] =  123	
	
The code creates a new DenseGraph with 10 vertices, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that numpy.array vertices can be added to a DenseGraph using the VertexList class in the constructor.  Finally, the first and second vertices are initialised with "abc" and 123 respectively

Methods 
-------
.. autoclass:: apgl.graph.DenseGraph.DenseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
