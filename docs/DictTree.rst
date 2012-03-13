DictTree
=========

A tree stucture with adjacencies stored in a dictionary based on DictGraph. A tree is essentially a directed graph in which all vertices (except for the root) can only have one incoming edge. The below code is a simple example showing how a tree is created and the output is the number of edges in the tree. 

::

        dictTree = DictTree()

        dictTree.addEdge("a", "b")
        dictTree.addEdge("a", "c")
        dictTree.addEdge("d", "a")
        
        print(dictTree.depth())

Methods 
-------
.. autoclass:: apgl.graph.DictTree.DictTree
   :members:
   :inherited-members:
   :undoc-members:
