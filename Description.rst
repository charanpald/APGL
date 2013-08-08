Another Python Graph Library is a simple, fast and easy to use graph library. Here is an example of its usage:

::

    >>> from apgl.graph import GeneralVertexList, SparseGraph 
    >>> numVertices = 5
    >>> graph = SparseGraph(numVertices)
    >>> graph[0,1] = 1
    >>> graph[0,2] = 3
    >>> graph[1,2] = 0.1
    >>> graph[3,4] = 2
    >>> graph.setVertex(0, "abc") 
    >>> graph.findConnectedComponents()
    [[0, 1, 2], [3, 4]]
    >>> graph.getWeightMatrix()
    array([[ 0. ,  1. ,  3. ,  0. ,  0. ],
           [ 1. ,  0. ,  0.1,  0. ,  0. ],
           [ 3. ,  0.1,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  2. ],
           [ 0. ,  0. ,  0. ,  2. ,  0. ]])
    >>> graph.degreeDistribution()
    array([0, 2, 3])
    >>> graph.neighbours(0)
    array([2, 1], dtype=int32)
    >>> print(graph)
    SparseGraph: vertices 5, edges 4, undirected, vertex list type: GeneralVertexList


More Information 
----------------

* See the user guide at http://pythonhosted.org/apgl/
* The source code is available at https://github.com/charanpald/APGL