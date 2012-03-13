DictGraph
===========

A graph with nodes stored in a dictionary. In particular the graph data structure is a dict of dicts and edges and vertices can be labeled with anything. This class is useful because unlike the graphs represented using adjacency/weight matrices, one can efficiently add vertices. For example: 

::

	from apgl.graph import * 

        graph = DictGraph()
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        graph["d", "e"] = 1
        graph.addEdge("d", "f", "abc")
        
In this code snippit, we construct an undirected graph and then add 5 edges to it with vertex labels as the strings a-e and edge labels 1. In the final line we have attached an edge label of "abc" to the edge going from "d" to "f". Note that one can easily take a DictGraph and copy the edges to one of the other graph classes: 


:: 

	from apgl.graph import * 
	
	graph = DictGraph()
        graph.addEdge("a", "b")
        graph.addEdge("a", "c")
        graph.addEdge("a", "d")
        
        edgeIndices = graph.getAllEdgeIndices() 
        
        graph2 = SparseGraph(GeneralVertexList(graph.getNumVertices())) 
        graph2.addEdges(edgeIndices) 
        
In this case the edges from graph are added to graph2 which is a SparseGraph object and represented using a scipy.sparse matrix. The corresponding mapping from vertex labels in graph and graph2 is found using getAllVertexIds. 

Methods 
-------
.. autoclass:: apgl.graph.DictGraph.DictGraph
   :members:
   :inherited-members:
   :undoc-members:
