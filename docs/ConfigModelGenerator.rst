ConfigModelGenerator
====================

A class to generate random graphs according to the Configuration Model. In this model, one specifies a particular degree sequence and a random graph is generated which fits the sequence as closely as possible. The following code generates a random graph with 10 vertices such that the ith vertex has degree i. 

::
	
	import numpy 
	from apgl.graph import * 
	from apgl.generator.ConfigModelGenerator import ConfigModelGenerator
	
	degreeSequence = numpy.arange(10)
        graph = SparseGraph(VertexList(10, 1))
        generator = ConfigModelGenerator(degreeSequence)
        graph = generator.generate(graph)

Methods 
-------
.. autoclass:: apgl.generator.ConfigModelGenerator.ConfigModelGenerator
   :members:
   :inherited-members:
   :undoc-members:
