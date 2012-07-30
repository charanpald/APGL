
import numpy 
from apgl.graph.VertexList import VertexList
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter 

class HIVVertices(VertexList):
    def __init__(self, numVertices):
        numFeatures = 8
        super(HIVVertices, self).__init__(numVertices, numFeatures)

        #Need to randomly set up the initial values
        self.V[:, self.dobIndex] = numpy.random.rand(numVertices)
        self.V[:, self.genderIndex] = Util.randomChoice(numpy.array([1, 1]), numVertices)
        #Note in reality females cannot be recorded as bisexual but we model the real scenario
        self.V[:, self.orientationIndex] = Util.randomChoice(numpy.array([4, 1]), numVertices)

        self.V[:, self.stateIndex] = numpy.zeros(numVertices)
        self.V[:, self.infectionTimeIndex] = numpy.ones(numVertices)*-1
        self.V[:, self.detectionTimeIndex] = numpy.ones(numVertices)*-1
        self.V[:, self.detectionTypeIndex] = numpy.ones(numVertices)*-1

        self.V[:, self.hiddenDegreeIndex] = numpy.ones(numVertices)*-1

    def setInfected(self, vertexInd, time):
        Parameter.checkIndex(vertexInd, 0, self.getNumVertices())
        Parameter.checkFloat(time, 0.0, float('inf'))

        if self.V[vertexInd, HIVVertices.stateIndex] == HIVVertices.infected:
            raise ValueError("Person is already infected")

        self.V[vertexInd, HIVVertices.stateIndex] = HIVVertices.infected
        self.V[vertexInd, HIVVertices.infectionTimeIndex] = time
        

    def setDetected(self, vertexInd, time, detectionType):
        Parameter.checkIndex(vertexInd, 0, self.getNumVertices())
        Parameter.checkFloat(time, 0.0, float('inf'))

        if detectionType not in [HIVVertices.randomDetect, HIVVertices.contactTrace]:
             raise ValueError("Invalid detection type : " + str(detectionType))

        if self.V[vertexInd, HIVVertices.stateIndex] != HIVVertices.infected:
            raise ValueError("Person must be infected to be detected")

        self.V[vertexInd, HIVVertices.stateIndex] = HIVVertices.removed
        self.V[vertexInd, HIVVertices.detectionTimeIndex] = time
        self.V[vertexInd, HIVVertices.detectionTypeIndex] = detectionType


    def copy(self):
        """
        Returns a copy of this object. 
        """
        vList = HIVVertices(self.V.shape[0])
        vList.setVertices(numpy.copy(self.V))
        return vList

    #Some static variables
    dobIndex = 0
    genderIndex = 1
    orientationIndex = 2

    #Time varying features
    stateIndex = 3
    infectionTimeIndex = 4
    detectionTimeIndex = 5
    detectionTypeIndex = 6
    hiddenDegreeIndex = 7

    male = 0
    female = 1
    
    hetero = 0
    bi = 1
    
    susceptible = 0
    infected = 1
    removed = 2
    randomDetect = 0
    contactTrace = 1 