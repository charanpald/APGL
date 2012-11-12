"""
Some methods which are useful but didn't quite fit into the graph classes. 

"""
import logging 
import numpy
import scipy.sparse 

class GraphUtils(object):
    def __init__(self):
        pass

    @staticmethod 
    def vertexLabelPairs(graph, edgeArray):
        """
        Create an array of pairs of vertex labels for each edge in the graph. Returns
        a tuple of this array and a vector of corresponding edge weights, for the
        indices of the full edge index array as given by getAllEdges.

        :param edgeArray: A numpy array with 2 columns, and values corresponding to vertex indices.
        :type edgeArray: :class:`numpy.ndarray`

        """
        #Parameter.checkList(edgeArray, Parameter.checkIndex, (0, self.getNumVertices()))

        numFeatures = graph.getVertexList().getNumFeatures()
        X = numpy.zeros((edgeArray.shape[0], numFeatures*2))

        for i in range(edgeArray.shape[0]):
            X[i, 0:numFeatures] = graph.getVertex(edgeArray[i, 0])
            X[i, numFeatures:numFeatures*2] = graph.getVertex(edgeArray[i, 1])

        return X 

    @staticmethod
    def vertexLabelExamples(graph):
        """
        Return a set of examples with pairs of vertex labels connected by an
        edge. For undircted graphs there exists an example (v_i, v_j) for
        every (v_j, v_i). Also, there is a set of negative examples where the
        edge does not exist. 
        """
        numFeatures = graph.getVertexList().getNumFeatures()
        numEdges = graph.getNumEdges()

        #Also add non-edges
        logging.info("Computing graph complement")
        cGraph = graph.complement()
        logging.info("Done with " + str(cGraph.getNumEdges()) + " edges.")
        perm = numpy.random.permutation(cGraph.getNumEdges())[0:numEdges]

        X = GraphUtils.vertexLabelPairs(graph, graph.getAllEdges())
        Xc = GraphUtils.vertexLabelPairs(cGraph, cGraph.getAllEdges()[perm, :])
        X = numpy.r_[X, Xc]

        y = numpy.ones(numEdges*2)
        y[numEdges:numEdges*2] = -1
        logging.debug(y)

        #If the graph is undirected add reverse edges
        if graph.isUndirected():
            X2 = numpy.zeros((numEdges*2, numFeatures*2))
            X2[:, 0:numFeatures] = X[:, numFeatures:numFeatures*2]
            X2[:, numFeatures:numFeatures*2] = X[:, 0:numFeatures]
            X = numpy.r_[X, X2]
            y = numpy.r_[y, y]

        return X, y

    @staticmethod
    def treeRoot(treeGraph):
        """
        Find the root of the given tree
        """
        inDegSeq = treeGraph.inDegreeSequence()
        root = numpy.nonzero(inDegSeq==0)[0][0]
        return root 

    @staticmethod 
    def treeDepth(treeGraph):
        """
        Find the depth of the given tree. 
        """
        if treeGraph.getNumVertices()==0:
            return 0 

        if not treeGraph.isTree():
            raise ValueError("Input graph is not a tree")
        
        root = GraphUtils.treeRoot(treeGraph)
        distances = treeGraph.dijkstrasAlgorithm(root)
        return numpy.max(distances[distances!=float('inf')])


    @staticmethod
    def modularity(W, clustering):
        """
        Give a symmetric weight matrix W and a clustering array "clustering", compute the
        modularity of Newman and Girvan. The input matrix W can be either an
        ndarray or a scipy.sparse matrix.
        """
        numVertices = W.shape[0]        
        clusterIds = numpy.unique(clustering)

        if type(W) == numpy.ndarray:
            degSequence = numpy.sum(W, 0)
        else:
            degSequence = numpy.array(W.sum(0)).ravel()

        numEdges = numpy.sum(degSequence)/2.0
        Q = 0

        for i in clusterIds:
            inds = numpy.arange(numVertices)[i==clustering]
            subW = W[inds, :][:, inds]

            Q += subW.sum()
            Q -= degSequence[inds].sum()**2/(2.0*numEdges)

        Q = Q/(2*numEdges)
        return Q 


    @staticmethod
    def kwayNormalisedCut(W, clustering):
        """
        Do k-way normalised cut. Each cluster should have at least 1 edge. The input
        matrix W can be either an ndarray or a scipy.sparse matrix.
        """
        numVertices = W.shape[0]
        clusterIds = numpy.unique(clustering)

        Q = 0 
        for i in clusterIds:
            inds = numpy.arange(numVertices)[i==clustering]
            invInds = numpy.arange(numVertices)[i!=clustering]
            numClustEdges = float((W[inds, :]).sum())
            if (len(invInds) != 0) and (numClustEdges != 0): 
                Q += (W[inds, :][:, invInds]).sum()/numClustEdges
            
        Q = Q/clusterIds.shape[0]
        return Q 


    @staticmethod 
    def shiftLaplacian(W):
        """
        Give a scipy sparse csr matrix W, compute the shifted Laplacian matrix,
        which is defined as I + D^-0.5 W D^-0.5 where D is a diagonal matrix of
        degrees. For vertices of degree zero, the corresponding row/col of the
        Laplacian is zero with a 0 at the diagonal. The eigenvalues of the shift
        Laplacian are between 0 and 2.
        """
        if not scipy.sparse.isspmatrix_csr(W):
            raise ValueError("W is not a csr matrix")
            
        W = scipy.sparse.csr_matrix(W, dtype=numpy.float)

        d = numpy.array(W.sum(0)).ravel()
        d[d!=0] = d[d!=0]**-0.5
        D = scipy.sparse.spdiags(d, 0, d.shape[0], d.shape[0], format='csr')
        
        i = numpy.zeros(W.shape[0])
        i[d!=0] = 1
        I = scipy.sparse.spdiags(i, 0, i.shape[0], i.shape[0], format='csr')
        
        Lhat = I + D.dot(W).dot(D)
        return Lhat
        
    @staticmethod 
    def normalisedLaplacianSym(W):
        """
        Give a scipy sparse csr matrix W, compute the normalised Laplacian matrix,
        which is defined as I - D^-0.5 W D^-0.5 where D is a diagonal matrix of
        degrees. For vertices of degree zero, the corresponding row/col of the
        Laplacian is zero with a 0 at the diagonal. The eigenvalues of the 
        Laplacian are between 0 and 2.
        """
        if not scipy.sparse.isspmatrix_csr(W):
            raise ValueError("W is not a csr matrix")
        W = scipy.sparse.csr_matrix(W, dtype=numpy.float)
        d = numpy.array(W.sum(0)).ravel()
        d[d!=0] = d[d!=0]**-0.5
        D = scipy.sparse.spdiags(d, 0, d.shape[0], d.shape[0], format='csr')
        
        i = numpy.zeros(W.shape[0])
        i[d!=0] = 1
        I = scipy.sparse.spdiags(i, 0, i.shape[0], i.shape[0], format='csr')
        
        Lhat = I - D.dot(W).dot(D)
        return Lhat        
    
    @staticmethod 
    def normalisedLaplacianRw(W): 
        """
        Compute the random walk Laplacian matrix given by D^-1 L where L is the 
        unnormalised Laplacian. 
        """
        if not scipy.sparse.isspmatrix_csr(W):
            raise ValueError("W is not a csr matrix")
            
        d = numpy.array(W.sum(0)).ravel()
        d[d!=0] = d[d!=0]**-1
        D = scipy.sparse.spdiags(d, 0, d.shape[0], d.shape[0], format='csr')
        
        i = numpy.zeros(W.shape[0])
        i[d!=0] = 1
        I = scipy.sparse.spdiags(i, 0, i.shape[0], i.shape[0], format='csr')
        
        Lhat = I - D.dot(W)
        return Lhat  
        
    
    @staticmethod
    def randIndex(clustering1, clustering2):
        """
        Compute the rand index for 2 clusterings given in arrays v1 and v2.
        """
        numVertices = clustering1.shape[0]
        error = 0        
        
        for i in range(numVertices):
            same_cl = clustering1[i] == clustering1
            same_learned_cl = clustering2[i] == clustering2
            error += (same_cl != same_learned_cl).sum()
        
        return float(error)/(numVertices*(numVertices-1))
        