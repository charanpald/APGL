library(RUnit)
source("Util.R")

test.generateData <- function() { 
  numExamples = 10 
  numFeatures = 20 
  
  X = generateData(numExamples, numFeatures)
  checkTrue(is.data.frame(X))
  checkEquals(nrow(X), numExamples)
  checkEquals(ncol(X), numFeatures+1)
  }

test.generateData2 <- function() { 
  numExamples = 10 
  numRelFeatures = 20 
  numIrrelFeatures = 10 
  
  X = generateData2(numExamples, numIrrelFeatures, numRelFeatures)
  checkTrue(is.data.frame(X))
  checkEquals(nrow(X), numExamples)
  checkEquals(ncol(X), numRelFeatures+numIrrelFeatures+1)
  }

test.bError <- function() { 
  yTrue = c(1, 1, 2, 2, 2)
  yPred = c(2, 1, 2, 2, 1)

  checkEquals(bError(yTrue, yPred), 2)
  checkEquals(bError(yTrue, yTrue), 0)
  checkEquals(bError(yPred, yPred), 0)
  
  yTrue = yTrue-1
  yPred = yPred-1
  
  checkEquals(bError(yTrue, yPred), 2)
  checkEquals(bError(yTrue, yTrue), 0)
  checkEquals(bError(yPred, yPred), 0)
  }

test.stratifiedCV <- function() { 
  Y = c(1, 1, 2, 2, 2, 2)
  bestresponse = 2
  folds = 2 
  
  sam = stratifiedCV(Y, bestresponse, folds)
  
  X = as.data.frame(table(Y))
  checkEquals(X[1, 2], 2)
  checkEquals(X[2, 2], 4)
  
  #Test with all ones 
  Y = c(2, 2, 2, 2, 2, 2)
  sam = stratifiedCV(Y, bestresponse, folds)
  
  print(sam)
  }
