"""
Some code to wrap the C++ code for graph matching in GraphM
""" 
import subprocess
import numpy 
import os 
import sys 
import tempfile
import logging 
from apgl.graph import SparseGraph, VertexList
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.data.Standardiser import Standardiser 

class GraphMatch(object): 
    def __init__(self, algorithm="PATH", alpha=0.5, featureInds=None, useWeightM=True):
        """
        Intialise the matching object with a given algorithm name, alpha 
        which is a trade of between matching adjacency matrices and vertex labels, 
        and featureInds which is an option array of indices to use for label 
        matching. 
        
        :param alpha: A value in [0, 1] which is smaller to match graph structure, larger to match the labels more  
        """
        Parameter.checkFloat(alpha, 0.0, 1.0)
        Parameter.checkClass(algorithm, str)
        
        self.algorithm = algorithm 
        self.alpha = alpha 
        self.maxInt = 10**9 
        self.featureInds = featureInds 
        self.useWeightM = useWeightM 
        #Gamma is the same as dummy_nodes_c_coef for costing added vertex labels         
        self.gamma = 0.0
        #Same as dummy_nodes_fill 
        self.rho = 0.5 
        
    def match(self, graph1, graph2): 
        """
        Take two graphs are match them. The two graphs must be AbstractMatrixGraphs 
        with VertexLists representing the vertices.  
        
        :param graph1: A graph object 
        
        :param graph2: The second graph object to match 
        
        :return permutation: A vector of indices representing the matching of elements of graph1 to graph2 
        :return distance: The graph distance list [graphDistance, fDistance, fDistanceExact] 
        """
        #Deal with case where at least one graph is emty 
        if graph1.size == 0 and graph2.size == 0: 
            permutation = numpy.array([], numpy.int)
            distanceVector = [0, 0, 0]  
            time = 0 
            return permutation, distanceVector, time 
        elif graph1.size == 0 or graph2.size == 0: 
            if graph1.size == 0: 
                graph1 = SparseGraph(VertexList(graph2.size, graph2.getVertexList().getNumFeatures()))
            else: 
                graph2 = SparseGraph(VertexList(graph1.size, graph1.getVertexList().getNumFeatures()))        
        
        numTempFiles = 5
        tempFileNameList = []         
        
        for i in range(numTempFiles): 
            fileObj = tempfile.NamedTemporaryFile(delete=False)
            tempFileNameList.append(fileObj.name) 
            fileObj.close() 
               
        configFileName = tempFileNameList[0]
        graph1FileName = tempFileNameList[1]
        graph2FileName = tempFileNameList[2]
        similaritiesFileName = tempFileNameList[3]
        outputFileName = tempFileNameList[4]

        if self.useWeightM:         
            W1 = graph1.getWeightMatrix()
            W2 = graph2.getWeightMatrix()
        else: 
            W1 = graph1.adjacencyMatrix()
            W2 = graph2.adjacencyMatrix()
        
        numpy.savetxt(graph1FileName, W1, fmt='%.5f')
        numpy.savetxt(graph2FileName, W2, fmt='%.5f')
        
        #Compute matrix similarities 
        C = self.vertexSimilarities(graph1, graph2)
        numpy.savetxt(similaritiesFileName, C, fmt='%.5f')
        
        #Write config file 
        configFile = open(configFileName, 'w')
        
        configStr = "graph_1=" + graph1FileName + " s\n"
        configStr +="graph_2=" + graph2FileName + " s\n"
        configStr +="C_matrix=" + similaritiesFileName + " s\n"
        configStr +="algo=" + self.algorithm + " s\n"
        configStr +="algo_init_sol=unif s\n"
        configStr +="alpha_ldh=" + str(self.alpha) + " d\n"
        configStr +="cdesc_matrix=A c\n"
        configStr +="cscore_matrix=A c\n"
        configStr +="hungarian_max=10000 d\n"
        configStr +="algo_fw_xeps=0.01 d\n"
        configStr +="algo_fw_feps=0.01 d\n"
        configStr +="dummy_nodes=0 i\n"
        configStr +="dummy_nodes_fill=" + str(self.rho) + " d\n"
        configStr +="dummy_nodes_c_coef=" + str(self.gamma) + " d\n"
        configStr +="qcvqcc_lambda_M=10 d\n"
        configStr +="qcvqcc_lambda_min=1e-5 d\n"
        configStr +="blast_match=0 i\n"
        configStr +="blast_match_proj=0 i\n"
        configStr +="exp_out_file=" + outputFileName + " s\n"
        configStr +="exp_out_format=Compact Permutation s\n"
        configStr +="verbose_mode=0 i\n"
        configStr +="verbose_file=cout s\n"
        
        configFile.write(configStr)
        configFile.close()
        
        fnull = open(os.devnull, 'w')
        #This is a bit hacky 
        try: 
            argList = ["/home/charanpal/.local/bin/graphm", configFileName] 
            subprocess.call(argList, stdout = fnull, stderr = fnull)    
        except OSError: 
            argList = ["/home/dhanjalc/.local/bin/graphm", configFileName] 
            subprocess.call(argList, stdout = fnull, stderr = fnull)
        fnull.close()
        
        #Next: parse input files 
        outputFile = open(outputFileName, 'r')
        
        line = outputFile.readline()        
        line = outputFile.readline() 
        line = outputFile.readline() 
        line = outputFile.readline() 
        
        graphDistance = float(outputFile.readline().split()[2]) 
        fDistance = float(outputFile.readline().split()[2])
        fDistanceExact = float(outputFile.readline().split()[2])
        time = float(outputFile.readline().split()[1]) 
        
        line = outputFile.readline() 
        line = outputFile.readline() 
        
        permutation = numpy.zeros(max(graph1.getNumVertices(), graph2.getNumVertices()), numpy.int)
        
        i = 0 
        for line in outputFile:
            permutation[i] = int(line.strip())-1
            i += 1
        
        #Delete files 
        os.remove(graph1FileName)
        os.remove(graph2FileName)
        os.remove(similaritiesFileName)
        os.remove(configFileName)
        os.remove(outputFileName)

        distanceVector = [graphDistance, fDistance, fDistanceExact]     
        return permutation, distanceVector, time 
        
    def vertexSimilarities(self, graph1, graph2): 
        """
        Compute a vertex similarity matrix C, such that the ijth entry is the matching 
        score between V1_i and V2_j, where larger is a better match. 
        """        
        if graph1.size == 0 and graph2.size == 0: 
            return numpy.zeros((graph1.size, graph2.size)) 
        
        if self.featureInds == None: 
            V1 = graph1.vlist.getVertices()
            V2 = graph2.vlist.getVertices()
        else: 
            V1 = graph1.vlist.getVertices()[:, self.featureInds]
            V2 = graph2.vlist.getVertices()[:, self.featureInds]
        
        return self.matrixSimilarity(V1, V2)
     
    def matrixSimilarity(self, V1, V2): 
        """
        Compute a vertex similarity matrix C, such that the ijth entry is the matching 
        score between V1_i and V2_j, where larger is a better match. 
        """  
        X = numpy.r_[V1, V2]
        standardiser = Standardiser()
        X = standardiser.normaliseArray(X)
        
        V1 = X[0:V1.shape[0], :]
        V2 = X[V1.shape[0]:, :]
        
        #print(X)
         
        #Extend arrays with zeros to make them the same size
        #if V1.shape[0] < V2.shape[0]: 
        #    V1 = Util.extendArray(V1, V2.shape, numpy.min(V1))
        #elif V2.shape[0] < V1.shape[0]: 
        #    V2 = Util.extendArray(V2, V1.shape, numpy.min(V2))
          
        #Let's compute C as the distance between vertices 
        #Distance is bounded by 1
        D = Util.distanceMatrix(V1, V2)
        maxD = numpy.max(D)
        minD = numpy.min(D)
        if (maxD-minD) != 0: 
            C = (maxD - D)/(maxD-minD)
        else: 
            C = numpy.ones((V1.shape[0], V2.shape[0])) 
            
        return C
     
    def distance(self, graph1, graph2, permutation, normalised=False, nonNeg=False, verbose=False):
        """
        Compute the graph distance metric between two graphs given a permutation 
        vector. This is given by F(P) = (1-alpha)/(||W1||^2_F + ||W2||^2_F)(||W1 - P W2 P.T||^2_F)
        - alpha 1/||C||_F tr(C.T P) in the normalised case. If we want an unnormalised 
        solution it is computed as (1-alpha)/(||W1 - P W2 P.T||^2_F) - alpha tr C.T P 
        and finally there is a standardised case in which the distance is between 
        0 and 1, where ||C||_F is used to normalise the vertex similarities and 
        we assume 0 <= C_ij <= 1. 
        
        :param graph1: A graph object 
        
        :param graph2: The second graph object to match 
        
        :param permutation: An array of permutation indices matching the first to second graph 
        :type permutation: `numpy.ndarray`
        
        :param normalised: Specify whether to normalise the objective function 
        :type normalised: `bool`
        
        :param nonNeg: Specify whether we want a non-negative solution.  
        :type nonNeg: `bool`
        
        :param verbose: Specify whether to return graph and label distance 
        :type nonNeg: `bool`
        """
        if graph1.size == 0 and graph2.size == 0: 
            if not verbose: 
                return 0.0
            else: 
                return 0.0, 0.0, 0.0
        elif graph1.size == 0 or graph2.size == 0: 
            if normalised: 
                if not verbose: 
                    return 1-self.alpha
                else: 
                    return 1-self.alpha, 1-self.alpha, 0.0
            else: 
                raise ValueError("Unsupported case")
        
        if self.useWeightM:         
            W1 = graph1.getWeightMatrix()
            W2 = graph2.getWeightMatrix()
        else: 
            W1 = graph1.adjacencyMatrix()
            W2 = graph2.adjacencyMatrix()
        
        if W1.shape[0] < W2.shape[0]: 
            W1 = Util.extendArray(W1, W2.shape, self.rho)
        elif W2.shape[0] < W1.shape[0]:
            W2 = Util.extendArray(W2, W1.shape, self.rho)
        
        n = W1.shape[0]
        P = numpy.zeros((n, n)) 
        P[(numpy.arange(n), permutation)] = 1
        dist1 = numpy.linalg.norm(W1 - P.dot(W2).dot(P.T))**2
        
        #Now compute the vertex similarities trace         
        C = self.vertexSimilarities(graph1, graph2)
        minC = numpy.min(C)
        maxC = numpy.max(C)
        C = Util.extendArray(C, (n, n), minC + self.gamma*(maxC-minC))

        dist2 = numpy.trace(C.T.dot(P))
        
        if normalised: 
            norm1 = ((W1**2).sum() + (W2**2).sum())
            norm2 = numpy.linalg.norm(C) 
            if norm1!= 0: 
                dist1 = dist1/norm1
            if norm2!= 0:
                dist2 = dist2/norm2
        
        dist = (1-self.alpha)*dist1 - self.alpha*dist2
        
        #If nonNeg = True then we add a term to the distance to ensure it is 
        #always positive. The numerator is an upper bound on tr(C.T P)
        if nonNeg and normalised:
            normC = norm2
    
            logging.debug("Graph distance: " + str(dist1) + " label distance: " + str(dist2) + " distance offset: " + str(self.alpha*n/normC) + " graph sizes: " + str((graph1.size, graph2.size)))           

            if normC != 0: 
                dist = dist + self.alpha*n/normC 
        else: 
            logging.debug("Graph distance: " + str(dist1) + " label distance: " + str(dist2) + " weighted distance: " + str(dist) + " graph sizes: " + str((graph1.size, graph2.size)))   
        
        if verbose: 
            return dist, dist1, dist2
        else: 
            return dist 
        
    def distance2(self, graph1, graph2, permutation):
        """
        Compute a graph distance metric between two graphs give a permutation 
        vector. This is given by F(P) = (1-alpha)/(||W1||^2_F + ||W2||^2_F)
        (||W1 - P W2 P.T||^2_F) - alpha 1/(||V1||_F^2 + ||V2||_F^2) ||V1 - P.T V2||^2_F 
        and is bounded between 0 and 1. 
        
        :param graph1: A graph object 
        
        :param graph2: The second graph object to match 
        
        :param permutation: An array of permutation indices matching the first to second graph 
        :type permutation: `numpy.ndarray`
        
        """
        if self.useWeightM:         
            W1 = graph1.getWeightMatrix()
            W2 = graph2.getWeightMatrix()
        else: 
            W1 = graph1.adjacencyMatrix()
            W2 = graph2.adjacencyMatrix()
        
        if W1.shape[0] < W2.shape[0]: 
            W1 = Util.extendArray(W1, W2.shape)
        elif W2.shape[0] < W1.shape[0]:
            W2 = Util.extendArray(W2, W1.shape)
        
        n = W1.shape[0]
        P = numpy.zeros((n, n)) 
        P[(numpy.arange(n), permutation)] = 1
        dist1 = numpy.linalg.norm(W1 - P.dot(W2).dot(P.T))**2
        
        #Now compute the vertex similarities distance         
        V1 = graph1.getVertexList().getVertices()
        V2 = graph2.getVertexList().getVertices()
        
        if V1.shape[0] < V2.shape[0]: 
            V1 = Util.extendArray(V1, V2.shape)
        elif V2.shape[0] < V1.shape[0]: 
            V2 = Util.extendArray(V2, V1.shape)
        
        dist2 = numpy.sum((V1 - P.T.dot(V2))**2)

        norm1 = ((W1**2).sum() + (W2**2).sum())
        norm2 = ((V1**2).sum() + (V2**2).sum())
        
        if norm1!= 0: 
            dist1 = dist1/norm1
        if norm2!= 0:
            dist2 = dist2/norm2         
        
        dist = (1-self.alpha)*dist1 + self.alpha*dist2
        
        return dist 
        
