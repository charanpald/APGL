
"""
Let's generate a simple dataset using uniform distributions and see how the
decision tree works with the data. 
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy 
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.util.Evaluator import Evaluator 

numExamples = 5000 
numFeatures = 2 

X = numpy.random.rand(numExamples, numFeatures)
y = numpy.zeros(numExamples)
noise = 0.05 

for i in range(numExamples): 
    if X[i, 1] > 0.5: 
        if X[i, 0] > 0.6: 
            y[i] = 0.78
        else: 
            y[i] = 0.65 
    else: 
        if X[i, 0] > 0.1: 
            y[i] = 0.74
        else: 
            if X[i, 1] < 0.1: 
                y[i] = 0.32
            else: 
                y[i] = 0.38 

y += numpy.random.randn(numExamples)*noise

numTrainExamples = numExamples*0.1 
numValidExamples = numExamples*0.1

trainX = X[0:numTrainExamples, :]
trainY = y[0:numTrainExamples]
validX = X[numTrainExamples:numTrainExamples+numValidExamples, :]
validY = y[numTrainExamples:numTrainExamples+numValidExamples]
testX = X[numTrainExamples+numValidExamples:, :]
testY = y[numTrainExamples+numValidExamples:]

learner = DecisionTreeLearner(minSplit=1, maxDepth=50)
learner.learnModel(trainX, trainY)


#Seem to be optimal 
alphaThreshold = 100
learner.prune(validX, validY, alphaThreshold)
#learner.tree = learner.tree.cut(3)

predY = learner.predict(testX)

plt.figure(0)
plt.scatter(testX[:, 0], testX[:, 1], c=testY, s=50, vmin=0, vmax=1)
plt.colorbar()

plt.figure(1)
plt.scatter(testX[:, 0], testX[:, 1], c=predY, s=50, vmin=0, vmax=1)
plt.colorbar()

colormap  = matplotlib.cm.get_cmap()

def displayTree(learner, vertexId, minX0, maxX0, minX1, maxX1, colormap): 
    vertex = learner.tree.getVertex(vertexId)
    if learner.tree.isLeaf(vertexId):
        p = mpatches.Rectangle([minX0, minX1], maxX0-minX0, maxX1-minX1, facecolor=colormap(vertex.getValue()), edgecolor="black")
        plt.gca().add_patch(p)            

    leftChildId = learner.getLeftChildId(vertexId)
        
    if learner.tree.vertexExists(leftChildId):
        if vertex.getFeatureInd() == 0: 
            displayTree(learner, leftChildId, minX0, vertex.getThreshold(), minX1, maxX1, colormap)
        else: 
            displayTree(learner, leftChildId, minX0, maxX0, minX1, vertex.getThreshold(), colormap)
        
    rightChildId = learner.getRightChildId(vertexId)    
    
    if learner.tree.vertexExists(rightChildId):
        if vertex.getFeatureInd() == 0: 
            displayTree(learner, rightChildId, vertex.getThreshold(), maxX0, minX1, maxX1, colormap)
        else: 
            displayTree(learner, rightChildId, minX0, maxX0, vertex.getThreshold(), maxX1, colormap)
            
plt.figure(2)
#p = mpatches.Rectangle([0, 0], 0.5, 0.5, facecolor=(0, 0, 1), edgecolor="red")
rootId = learner.tree.getRootId()
displayTree(learner, rootId, 0, 1, 0, 1, colormap)



#Next plot test error versus alpha 
#First figure out range of alphas 
minAlpha = 100 
maxAlpha = -100

for vertexId in learner.tree.getAllVertexIds(): 
    alpha = learner.tree.getVertex(vertexId).alpha
    
    if alpha < minAlpha: 
        minAlpha = alpha 
    if alpha > maxAlpha: 
        maxAlpha = alpha 
        
numAlphas = 100
alphas = numpy.linspace(maxAlpha+0.1, minAlpha, numAlphas)
errors = numpy.zeros(numAlphas)

for i in range(alphas.shape[0]): 
    learner.prune(validX, validY, alphas[i])
    predY = learner.predict(testX)
    errors[i] = Evaluator.rootMeanSqError(predY, testY)
    
plt.figure(3)
plt.scatter(alphas, errors)

#Now plot best tree 
plt.figure(4)
learner.learnModel(trainX, trainY)
learner.prune(validX, validY, alphas[numpy.argmin(errors)])
rootId = learner.tree.getRootId()
displayTree(learner, rootId, 0, 1, 0, 1, colormap)

plt.show()
    
    
