
library(TreeRank)
library(RUnit)
source("MSLeafRanks.R")

#Test MyLeafRank functions 
testLRsvm2 <- function() { 
  numExamples = 100
  XY = generateData(numExamples=numExamples)             
  csvm <- LRsvm2(class~., XY, bestresponse = 1)
  
  #Now try actually running treerank 
  TreeRank(class~., XY, bestresponse = 1,LeafRank=LRsvm2)
  
  #Now try case where labels are all 1 
  XY[["class"]] = matrix(1, nrow = numExamples, ncol = 1 ) 
  
  csvm <- LRsvm2(class~., XY, bestresponse = 1)
  yPred = predict(csvm, XY)
  #print(yPred)
}

testLRsvmLinear <- function() { 
  numExamples = 100
  XY = generateData(numExamples=numExamples)           
  csvm <- LRsvmLinear(class~., XY, bestresponse = 1)
  
  #Now try actually running treerank 
  TreeRank(class~., XY, bestresponse = 1,LeafRank=LRsvmLinear)
  
  #Now try case where labels are all 1 
  XY[["class"]] = matrix(1, nrow = numExamples, ncol = 1 ) 
  
  csvm <- LRsvmLinear(class~., XY, bestresponse = 1)
  yPred = predict(csvm, XY)
  #print(yPred)
}

testLRCart2 <- function() { 
  XY = generateData()          
  cart <- LRCart2(class~., XY, bestresponse = 1)
  
  TreeRank(class~., XY, bestresponse = 1,LeafRank=LRCart2)
}

testLRforest2 <- function() { 
  XY = generateData()               
  cart <- LRforest2(class~., XY, bestresponse = 1)
  
  TreeRank(class~., XY, bestresponse = 1,LeafRank=LRforest2)
}

testLRCartF <- function() { 
  XY = generateData2(500, numIrrFeatures=0, numRelFeatures=900)      
  cart <- LRCartF(class~., XY, bestresponse = 3)
  yPred = predict(cart, XY)

  tree = TreeRank(class~., XY, bestresponse = 3, LeafRank=LRCartF)
  roc = getROC(tree, XY)
  print(roc)
  XY
}

testLRsvmF <- function() { 
  XY = generateData2(200, numIrrFeatures=0, numRelFeatures=900)      
  cart <- LRsvmF(class~., XY, bestresponse = 3)
  yPred = predict(cart, XY)

  tree = TreeRank(class~., XY, bestresponse = 3, LeafRank=LRsvmF)
  roc = getROC(tree, XY)

  
  XY
}

testLRCartF()
  