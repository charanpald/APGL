
rm(list = ls(all = TRUE))

#Predict using elastic net and wavelets 
#Compare to results in paper 
#Consider functional classification
 
saveAllResults <- function(fileName, maxNRMIndex, savePrefix) { 
  D = read.table(fileName, header=TRUE, row.names=1, sep=",")
  X <- D[,1:maxNRMIndex]
    
  saveResults(X, D, "Ind.Testo.1", savePrefix)
  saveResults(X, D, "Ind.Testo.2", savePrefix)
  saveResults(X, D, "Ind.Testo.3", savePrefix)
  saveResults(X, D, "Ind.Cortisol.1", savePrefix)
  saveResults(X, D, "Ind.Cortisol.2", savePrefix)
  saveResults(X, D, "Ind.Cortisol.3", savePrefix)
  saveResults(X, D, "Ind.IGF1.1", savePrefix)
  saveResults(X, D, "Ind.IGF1.2", savePrefix)
  saveResults(X, D, "Ind.IGF1.3", savePrefix)
  }

#Next, output ROC curves 
source("CrossValidation.R")
source("MSLeafRanks.R")

#Now produce results for all classification 
fileName = "../../../../data/metabolomic/data.RMN.total.6.txt"
savePrefix = "TR-RMN-"
maxNRMIndex = 950
saveAllResults(fileName, maxNRMIndex, savePrefix)

fileName = "data/data.sportsmen.log.AP.1.txt"
maxNRMIndex = 419
savePrefix = "TR-sportsmen-"
saveAllResults(fileName, maxNRMIndex, savePrefix)
  