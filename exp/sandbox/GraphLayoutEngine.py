"""
A class just to layout the nodes of a graph according to an adapted version of the
spring layout. Takes as imput a networkx graph
"""

import numpy

class GraphLayoutEngine():
    def __init__(self):
        self.attractK = 20.0
        self.repulseK = 20.0
        self.centerK = 20.0
        self.vertexDiameter = 0.1
        self.dim = 2
        self.maxIters = 200
        self.timeStep = 0.2
        self.damping = 0.1
        self.coolingFactor = 0.80

    """
    The vector of ages is used to apply a force towards the center. 
    """
    def layout(self, graph, ages=None):
        numVertices = graph.order()
        charges = numpy.ones((numVertices, 1), numpy.float64)
        velocities = numpy.zeros((numVertices, self.dim), numpy.float64)
        positions = numpy.random.rand(numVertices, self.dim, 1)


        jvec = numpy.ones((numVertices, 1))

        if ages==None:
            ages = numpy.zeros(numVertices)

        C = numpy.dot(charges, charges.T)
        C = numpy.reshape(C, (numVertices, 1, numVertices))

        #Create a dictionary which maps from indices to nodes
        indexNodeDict, nodeIndexDict = self.createIndexDict(graph)

        G = self.attractK/(numpy.outer((1-ages), (1-ages))+1)
        G2 = numpy.zeros((numVertices, self.dim, numVertices))

        for edge in graph.edges():
            i = nodeIndexDict[edge[0]]
            j = nodeIndexDict[edge[1]]
            G2[i, :, j] = G[i, j]
            G2[j, :, i] = G[i, j]

        tol = numVertices/10000.0
        maxVelocity = 500.0
        averageKe = tol + 1

        index = 0
        allPositions = {}

        while averageKe > tol and index < self.maxIters:
            #D[i,:,j] is diff in positions[i, :, 1] - positions[i, :, 1]
            #The problem is that D is n^2 (same with D2 and R) 
            P = numpy.dot(positions, jvec.T)
            D = P - P.T
            D2 = (numpy.sum(D**2, 1) + self.vertexDiameter)**1.5
            D2 = numpy.reshape(D2, (numVertices, 1, numVertices))
            R = self.repulseK*D*C / D2
            A = -D*G2

            netForces =  - self.centerK * numpy.array([ages]).T  * positions[:, :, 0]/numpy.array([numpy.sqrt(numpy.sum(positions[:, :, 0]**2, 1))]).T
            netForces[:, 0] = netForces[:, 0] + numpy.sum(R[:, 0, :], 1) + numpy.sum(A[:, 0, :], 1)
            netForces[:, 1] = netForces[:, 1] + numpy.sum(R[:, 1, :], 1) + numpy.sum(A[:, 1, :], 1)

            velocities = velocities*self.damping + (self.timeStep*self.damping)*netForces
            #velocities = velocities + self.timeStep*netForces
            #velMags = numpy.sum(velocities**2, 1)
            #velMagsMax = numpy.minimum(velMags, maxVelocity)
            #velocities = velocities*numpy.array([velMagsMax]).T/numpy.array([velMags]).T
            positions[:,:,0] = positions[:,:,0] + velocities*self.timeStep
            averageKe = numpy.sum(velocities**2)/numVertices


            maxVelocity = maxVelocity * self.coolingFactor
            allPositions[index] = self.createPositionsDict(graph, numpy.copy(positions))
            index += 1
            print(("Iteration " + str(index) + ", average Ke " + str(averageKe)))
            
        print("All done")
        positionsDict = self.createPositionsDict(graph, positions)

        return positionsDict, allPositions, index

    def createPositionsDict(self, graph, positions):
        positionsDict = {}
        index = 0

        for node in graph.nodes():
            positionsDict[node] = positions[index, :, 0]
            index = index + 1

        return positionsDict

    def createIndexDict(self, graph):
        index = 0
        indexNodeDict = {}
        nodeIndexDict = {}

        for node in graph.nodes():
            indexNodeDict[index] = node
            nodeIndexDict[node] = index
            index = index + 1

        return indexNodeDict, nodeIndexDict
