"""
Some code to wrap the C++ code for graph matching in GraphM
""" 
import subprocess
import numpy 
import os 
import sys 
import tempfile
from apgl.util.PathDefaults import PathDefaults
from apgl.data.Standardiser import Standardiser 
from apgl.kernel.LinearKernel import LinearKernel

class GraphMatch(object): 
    def __init__(self, algorithm="PATH", alpha=0.5):
        self.algorithm = algorithm 
        self.alpha = alpha 
        
        self.maxInt = 10**9 
        
    def match(self, graph1, graph2): 
        """
        Take two graphs are match them. The two graphs must be AbstractMatrixGraphs 
        with VertexLists representing the vertices.  
        """
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
        
        W1 = graph1.getWeightMatrix()
        numpy.savetxt(graph1FileName, W1, fmt='%.5f')
        
        W2 = graph2.getWeightMatrix()
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
        configStr +="dummy_nodes_fill=0 d\n"
        configStr +="dummy_nodes_c_coef=0.01 d\n"
        configStr +="qcvqcc_lambda_M=10 d\n"
        configStr +="qcvqcc_lambda_min=1e-5 d\n"
        configStr +="blast_match=1 i\n"
        configStr +="blast_match_proj=0 i\n"
        configStr +="exp_out_file=" + outputFileName + " s\n"
        configStr +="exp_out_format=Compact Permutation s\n"
        configStr +="verbose_mode=0 i\n"
        configStr +="verbose_file=cout s\n"
        
        configFile.write(configStr)
        configFile.close()
        
        #This is a bit hacky 
        try: 
            argList = ["/home/charanpal/local/bin/graphm", configFileName] 
            subprocess.call(argList)    
        except OSError: 
            argList = ["/home/dhanjalc/local/bin/graphm", configFileName] 
            subprocess.call(argList)  
        
        #Next: parse input files 
        outputFile = open(outputFileName, 'r')
        
        line = outputFile.readline()        
        line = outputFile.readline() 
        line = outputFile.readline() 
        line = outputFile.readline() 
        
        distance = float(outputFile.readline().split()[2]) 
        line = outputFile.readline() 
        line = outputFile.readline() 
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
         
        return permutation, distance, time 
        
    def vertexSimilarities(self, graph1, graph2): 
        V1 = graph1.getVertexList().getVertices()
        V2 = graph2.getVertexList().getVertices()
        
        V1 = Standardiser().normaliseArray(V1.T).T
        V2 = Standardiser().normaliseArray(V2.T).T
        
        C = LinearKernel().evaluate(V1, V2)
        
        return C 
        
    def distance(self, graph1, graph2, permutation): 
        W1 = graph1.getWeightMatrix()
        W2 = graph2.getWeightMatrix()
        
        if W1.shape[0] < W2.shape[0]: 
            tempW1 = numpy.zeros(W2.shape)
            tempW1[0:W1.shape[0], 0:W1.shape[0]] = W1
            W1 = tempW1 
        elif W2.shape[0] < W1.shape[0]:
            tempW2 = numpy.zeros(W1.shape)
            tempW2[0:W2.shape[0], 0:W2.shape[0]] = W2
            W2 = tempW2 
        
        n = W1.shape[0]
        P = numpy.zeros((n, n)) 
        #P[(permutation, numpy.arange(n))] = 1
        P[(numpy.arange(n), permutation)] = 1
        dist = numpy.linalg.norm(W1 - P.dot(W2).dot(P.T))**2
        
        #Now compute the vertex similarities trace         
        C = self.vertexSimilarities(graph1, graph2)
        
        if C.shape[0] != C.shape[1]: 
            n = max(C.shape[0], C.shape[1])
            tempC = numpy.ones((n, n))*C.min()
            tempC[0:C.shape[0], 0:C.shape[1]] = C
            C = tempC 
        
        dist2 = numpy.trace(C.T.dot(P))
        
        return dist 
        
        
        