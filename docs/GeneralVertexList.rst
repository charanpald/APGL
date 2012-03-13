GeneralVertexList
=================
The GeneralVertexList object represents an indexed list of vertices which may be anything, and is often used in conjection with either SparseGraph, PySparseGraph or DenseGraph. The following example demonstrates its usage: 

:: 

	from apgl.graph import * 
	import numpy 

	numVertices = 10 
	numFeatures = 3

	vList = GeneralVertexList(numVertices)
	vList.setVertex(0, "abc")
	vList.setVertex(1, "def")

This code creates a GeneralVertexList object, and assigns strings to the first and second vertices. 


Methods 
-------
.. autoclass:: apgl.graph.GeneralVertexList.GeneralVertexList
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
