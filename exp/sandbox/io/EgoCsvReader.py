'''
Created on 6 Aug 2009

@author: charanpal
'''
import logging
import numpy 
import numpy.random as rand
import csv
from apgl.data.ExamplesList import ExamplesList
from apgl.util.Util import Util
from apgl.io.CsvReader import CsvReader
from apgl.graph.SparseGraph import SparseGraph

class EgoCsvReader(CsvReader):
    '''
    Read a CSV file containing all the relevant features from the SPSS ego file. Removes all crap, and 
    reformats data as appropriate. 
    '''
    def __init__(self):
        self.numPossibleAlters = 15
        self.altersGap = 9 #Alters gap in the CSV files
        self.numProfessions = 8
        self.numOrganisationTypes = 10
        self.numDiscussions = 6
        self.numKnowledgeSources = 9
        self.partialAlterFields = 11
        self.printIterationStep = 100
        self.numSources = 9
        self.p = 0.7
        self.k = 3
        self.numTestQuestions = 12
        
        (self.egoQuestionIds, self.alterQuestionIds) = self.__getEgoAndAlterQuestionIds()

        #Store some common indices
        self.ageIndex = self.egoQuestionIds.index(("Q5X", 0)) #Index of the age (in years) in the Ego CSV files
        self.genderIndex = self.egoQuestionIds.index(("Q4", 0))
        self.educationIndex = self.egoQuestionIds.index(("Q48", 0))       
        self.incomeIndex = self.egoQuestionIds.index(("Q51X", 0))
        self.townSizeIndex = self.egoQuestionIds.index(("Q13X", 0))
        

        self.numFriendsIndex = self.egoQuestionIds.index(("Q44AX", 0))
        self.numColleaguesIndex = self.egoQuestionIds.index(("Q44BX", 0))
        self.numFamilyIndex = self.egoQuestionIds.index(("Q44CX", 0))
        self.numAquantancesIndex = self.egoQuestionIds.index(("Q44DX", 0))

        self.homophileAgeIndex = self.egoQuestionIds.index(("Q46A", 0))
        self.homophileGenderIndex = self.egoQuestionIds.index(("Q46B", 0))
        self.homophileEducationIndex = self.egoQuestionIds.index(("Q46C", 0))
        self.homophileIncomeIndex = self.egoQuestionIds.index(("Q46D", 0))

        
        self.foodRiskIndex = self.egoQuestionIds.index(("Q28A", 0))
        self.experienceIndex = self.egoQuestionIds.index(("Q22", 0))
        self.internetFreqIndex = self.egoQuestionIds.index(("Q55X", 0))
        self.peopleAtWorkIndex = self.egoQuestionIds.index(("Q50X", 0))

        (self.egoTestIds, self.alterTestIds) = self.getEgoAlterTestIds()
        

    """
    Given a number n of total contacts for a particular person, and a set
    H of homophiles and N of non-homophiles, we pick at most p*n from H
    and the remaining contacts are chosen from N.
    """
    def setP(self, p): 
        self.p = p 
    
    def getP(self):
        return p

    def __getEgoAndAlterQuestionIds2(self):
        """
        Just return gender, age, education and profession. 
        """
        egoQuestionIds = [("Q4",0), ("Q5X",0), ("Q48",0)]
        alterQuestionIds = [("Q184$",0), ("Q185$X",0), ("Q186$",0)]

        egoQuestionIds.extend([("Q51X",0)])
        alterQuestionIds.extend([("Q191$X",0)])

        for i in range(1, self.numProfessions+1):
            egoQuestionIds.append(("Q7_" + str(i), 1))
            alterQuestionIds.append(("Q187$_" + str(i) , 1))

        egoQuestionIds.extend([("Q44AX", 0), ("Q44BX", 0), ("Q44CX",0), ("Q44DX",0)])
        alterQuestionIds.extend([("Q180A$X", 0), ("Q180B$X", 0), ("Q180C$X",0), ("Q180D$X",0)])

        egoQuestionIds.extend([("Q46A", 0), ("Q46B", 0), ("Q46C", 0), ("Q46D", 0)])
        alterQuestionIds.extend([("Q182A$", 0), ("Q182B$", 0), ("Q182C$", 0), ("Q182D$", 0)])

        return (egoQuestionIds, alterQuestionIds)

    def __getEgoAndAlterQuestionIds(self):
        egoQuestionIds = [("Q4",0), ("Q5X",0), ("Q48",0)]
        alterQuestionIds = [("Q184$",0), ("Q185$X",0), ("Q186$",0)]

        for i in range(1, self.numProfessions+1):
            egoQuestionIds.append(("Q7_" + str(i), 1))
            alterQuestionIds.append(("Q187$_" + str(i) , 1))

        egoQuestionIds.extend([("Q51X",0), ("Q52",0), ("Q53X",0), ("Q13X",0), ("Q20A",0)])
        alterQuestionIds.extend([("Q191$X",0), ("Q192$",0), ("Q193$X",0), ("Q195$X",0), ("Q4A$",0)])

        egoQuestionIds.extend([("Q26X",0), ("Q28A",0), ("Q22",0), ("Q34", 0)])
        alterQuestionIds.extend([("Q24$X",0), ("Q1A$",0), ("Q2$",0), ("Q21$", 0)])

        egoQuestionIds.extend([("Q16#X",0)])
        alterQuestionIds.extend([("Q32$X",0)])

        egoQuestionIds.extend([("Q55X", 0), ("Q17bisA#X", 0)])
        alterQuestionIds.extend([("Q196$X", 0), ("Q33A$X", 0)])

        egoQuestionIds.extend([("Q37A", 0), ("Q37C", 0), ("Q37D", 0)])
        alterQuestionIds.extend([("Q20A$", 0), ("Q20C$", 0), ("Q20D$", 0)])

        egoQuestionIds.extend([("Q42A", 0), ("Q42B",0), ("Q42C",0), ("Q42D",0)])
        alterQuestionIds.extend([("Q178A$", 0), ("Q178B$",0), ("Q178C$",0), ("Q178D$",0)])
        
        for i in range(1, self.numOrganisationTypes+1):
            egoQuestionIds.append(("Q43M_" + str(i) , 1))
            alterQuestionIds.append(("Q179M$_" + str(i) , 1))

        egoQuestionIds.extend([("Q44AX", 0), ("Q44BX", 0), ("Q44CX",0), ("Q44DX",0), ("Q45", 0)])
        alterQuestionIds.extend([("Q180A$X", 0), ("Q180B$X", 0), ("Q180C$X",0), ("Q180D$X",0), ("Q181$", 0)])

        for i in range(1, self.numDiscussions+1):
            egoQuestionIds.append(("Q47CM_" + str(i) , 1))
            alterQuestionIds.append(("Q183CM$_" + str(i) , 1))

        for i in range(1, self.numDiscussions+1):
            egoQuestionIds.append(("Q47EM_" + str(i) , 1))
            alterQuestionIds.append(("Q183EM$_" + str(i) , 1))

        egoQuestionIds.extend([("Q50X", 0), ("Q46A", 0), ("Q46B", 0), ("Q46C", 0), ("Q46D", 0)])
        alterQuestionIds.extend([("Q190$X", 0), ("Q182A$", 0), ("Q182B$", 0), ("Q182C$", 0), ("Q182D$", 0)])
        
        return (egoQuestionIds, alterQuestionIds)
        
    
    def getEgoQuestionIds(self):
        return self.egoQuestionIds

    def getAlterQuestionIds(self):
        return self.alterQuestionIds

    def getEgoAlterTestIds(self):
        egoTestIds = []
        alterTestIds = [] 

        for i in range(1, self.numTestQuestions+1):
            egoTestIds.append(("Q10M" + str(i) + ".", 1))
            alterTestIds.append(("Q7M" + str(i) + ".", 1))

        return egoTestIds, alterTestIds 


    def findInfoDecayGraph(self, egoTestFileName, alterTestFileName, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR, egoFileName, alterFileName, missing=0):
        (egoTestArray, egoTitles) = self.readFile(egoTestFileName, self.egoTestIds, missing=0)
        (alterTestArray, alterTitles) = self.readFile(alterTestFileName, self.alterTestIds, missing=0)

        egoMarks = numpy.zeros(egoTestArray.shape[0])
        alterMarks = numpy.zeros(alterTestArray.shape[0])
        decays = numpy.zeros(egoIndicesR.shape[0])

        correctAnswers = numpy.array([1,2,4,5,8,9,10,12])
        wrongAnswers = numpy.array([3, 6, 7, 11])

        for i in range(egoTestArray.shape[0]):
            egoMarks[i] = numpy.intersect1d(egoTestArray[i], correctAnswers).shape[0]
            egoMarks[i] += wrongAnswers.shape[0] - numpy.intersect1d(egoTestArray[i], wrongAnswers).shape[0]

        for i in range(alterMarks.shape[0]):
            alterMarks[i] = numpy.intersect1d(alterTestArray[i], correctAnswers).shape[0]
            alterMarks[i] += wrongAnswers.shape[0] - numpy.intersect1d(alterTestArray[i], wrongAnswers).shape[0]

        """
        We just return how much the alter understood, since this represents the
        decay in understanding of the ego and transmission.
        """
        for i in range(decays.shape[0]):
            decays[i] = alterMarks[alterIndices[i]]

        #A lot of people could not be bothered to fill the questions and hence
        #get 4 correct by default 
        decays = (decays) / float(self.numTestQuestions)
        defaultDecay = 10**-6

        #Finally, put the data into a graph
        (egoArray, egoTitles) = self.readFile(egoFileName, self.egoQuestionIds, missing)
        (alterArray, alterTitles) = self.readFile(alterFileName, self.alterQuestionIds, missing)

        V = egoArray
        V = numpy.r_[V, alterArray[alterIndices, :]]
        V = numpy.r_[V, egoArray[alterIndicesNR, :]]
        vList = VertexList(V.shape[0], V.shape[1])
        vList.setVertices(V)
        graph = SparseGraph(vList, False)

        edgesR = numpy.c_[egoIndicesR, egoArray.shape[0]+numpy.arange(alterIndices.shape[0])]
        graph.addEdges(edgesR, decays)

        edgesNR = numpy.c_[egoIndicesNR, egoArray.shape[0]+alterIndices.shape[0]+numpy.arange(alterIndicesNR.shape[0])]
        graph.addEdges(edgesNR, numpy.ones(edgesNR.shape[0]) * defaultDecay)

        return graph


    #Here we read each line which represents an ego and then construct data 
    def readFiles(self, egoFileName, alterFileName, missing=0):
        (egoArray, egoTitles) = self.readFile(egoFileName, self.egoQuestionIds, missing)
        (alterArray, alterTitles) = self.readFile(alterFileName, self.alterQuestionIds, missing)

        #Augment receivers with new information
        egoAlterQuestionIds = self.__getAlterQuestionIds()
        alterFieldIndices = self.getAlterFieldIndices()

        (egoAlterArray, egoAlterTitles) = self.readFile(egoFileName, egoAlterQuestionIds, missing)
        (receiversArray, egoIndicesR, alterIndices) = self.generateReceivers(egoAlterArray, alterArray, alterFieldIndices)

        #Make sure we count receivers for all egos 
        receiverCounts = numpy.zeros(egoArray.shape[0], numpy.int)
        if egoIndicesR.shape[0] !=0:
            binCount = numpy.bincount(egoIndicesR)
        else:
            binCount = numpy.array([])
        receiverCounts[0:binCount.shape[0]] = binCount 

        #Generate non-receivers 
        numContactsIndices = [self.numFriendsIndex, self.numColleaguesIndex, self.numFamilyIndex, self.numAquantancesIndex]
        homophileIndexPairs = [(self.homophileAgeIndex, self.ageIndex), (self.homophileGenderIndex, self.genderIndex)]
        homophileIndexPairs.extend([(self.homophileEducationIndex, self.educationIndex), (self.homophileIncomeIndex, self.incomeIndex)])

        (nonReceiversArray, egoIndicesNR, alterIndicesNR) = self.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)
        
        #Now, we generate all pairs of senders/non-senders and receivers/non-receivers 
        numExamples = nonReceiversArray.shape[0] + receiversArray.shape[0]
        numPersonFeatures = egoArray.shape[1]
        numFeatures = numPersonFeatures*2
        
        X = numpy.zeros((numExamples, numFeatures))
        y = numpy.zeros(numExamples, numpy.int32)

        for i in range(0, numExamples): 
            if i < nonReceiversArray.shape[0]: 
                X[i, 0:numPersonFeatures] = egoArray[egoIndicesNR[i], :]
                X[i, numPersonFeatures:numFeatures] = nonReceiversArray[i, :]
                y[i] = -1
            else:
                j = i - nonReceiversArray.shape[0]
                X[i, 0:numPersonFeatures] = egoArray[egoIndicesR[j], :]
                X[i, numPersonFeatures:numFeatures] = receiversArray[j, :]
                y[i] = 1
                
        examplesList = ExamplesList(numExamples)
        examplesList.addDataField("X", X)
        examplesList.addDataField("y", y)
        examplesList.setDefaultExamplesName("X")
        examplesList.setLabelsName("y")

        return examplesList, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR 

    def generateReceivers(self, egoAlterArray, realAltersArray, alterFieldIndices):
        """ 
        Takes in a row for each ego with up to 15 egos on each line, extract alters and augment data. 
        The parameter alterFieldIndices is a list of indices in realAltersArray that match those present 
        in egoAlterArray. 
        """
        numEgos = egoAlterArray.shape[0]
        maxAlters = numEgos * self.numPossibleAlters
        
        generatedAltersArray = numpy.zeros((maxAlters, realAltersArray.shape[1]))
        egoIndices = numpy.zeros(maxAlters, numpy.int)
        alterIndices = numpy.zeros(maxAlters, numpy.int32)
        receiverIndex = 0 
        
        logging.info("Generating receivers for " + str(numEgos) + " egos")
        
        for i in range(0, numEgos): 
            Util.printIteration(i, self.printIterationStep, numEgos)
            
            for j in range(0, self.numPossibleAlters): 
                if egoAlterArray[i, j*self.partialAlterFields] != -1 and egoAlterArray[i, j*self.partialAlterFields] != 0:  
                    candidateAlters = numpy.array(list(range(0, realAltersArray.shape[0])))
                    
                    for k in range(0, len(alterFieldIndices)): 
                        subset = numpy.nonzero(realAltersArray[:, alterFieldIndices[k]] == egoAlterArray[i, j*self.partialAlterFields+k])[0]
                        candidateAlters = numpy.intersect1d(candidateAlters, subset)
                   
                    if candidateAlters.shape[0] != 0: 
                        alterIndices[receiverIndex] = candidateAlters[rand.randint(0, candidateAlters.shape[0])]
                        egoIndices[receiverIndex] = i
                        
                        chosenAlter = realAltersArray[alterIndices[receiverIndex], :]
                        generatedAltersArray[receiverIndex, :] = chosenAlter
                        receiverIndex = receiverIndex + 1 
                else: 
                    break 
                
        generatedAltersArray = generatedAltersArray[0:receiverIndex, :]
        egoIndices = egoIndices[0:receiverIndex]
        alterIndices = alterIndices[0:receiverIndex]
        logging.info("Done - chose " + str(receiverIndex) + " receivers")

        return (generatedAltersArray, egoIndices, alterIndices)
    
    def __getAlterQuestionIds(self):
        questionIds = []
        
        for i in range(0, self.numPossibleAlters): 
            questionIds.append(("Q" + str(26+i*self.altersGap) + "#", 2))
            questionIds.append(("Q" + str(27+i*self.altersGap) + "#", 2))

            for j in range(1, self.numProfessions+1):
                questionIds.append(("Q" + str(28+i*self.altersGap) + "#_" + str(j), 2))

            questionIds.append(("Q" + str(29+i*self.altersGap) + "#", 2))
            
        return questionIds

    def getAlterFieldIndices(self):
        alterFieldIndices = [self.genderIndex, self.ageIndex]

        for i in range(1, self.numProfessions+1):
            alterFieldIndices.append(self.egoQuestionIds.index(("Q7_" + str(i) , 1)))

        alterFieldIndices.append(self.educationIndex)

        return alterFieldIndices
    
    def readFile(self, fileName, questionIds, missing=0):
        """ 
        Read a CSV file of numbers and possibly some missing values. The first line of the file is a 
        list of titles for the columns. Returns an array of the numbers with missing values replaced 
        with zeros. 
        """
        numLines = self.getNumLines(fileName)

        try:
            reader = csv.reader(open(fileName, "rU"))
        except IOError:
            raise
        
        titles = next(reader)
        
        rowIndex = 0 
        numFields = len(questionIds)
        X = numpy.zeros((numLines-1, numFields))
    
        for row in reader:
            X[rowIndex, :] = self.csvRowToVector(row, questionIds, titles)
            rowIndex = rowIndex + 1 

        if missing == 0:
            logging.info("Keeping missing values as zero")
        elif missing == 1:
            logging.info("Replacing missing values with mean.")
            X = self.replaceMissingValues(X) 
        elif missing == 2:
            logging.info("Replacing missing values with N(mu, sigma^2).")
            X = self.replaceMissingValues2(X)
        elif missing == 3:
            logging.info("Replacing missing values with mode")
            X = self.replaceMissingValuesMode(X)
        else:
            raise ValueError("Invalid missing value treatment: " + missing)
        
        logging.info("Done - generated array with " + str(X.shape[0]) + " rows and " + str(X.shape[1]) + " columns")
        return X, titles
    
    def csvRowToVector(self, csvRow, questionIds, csvTitles):
        """
        Take a list of strings which are either numbers or empty strings (csvRow), and a list of pairs of 
        tuples (questionIds, hasMissingValues) and return a vector with missing values set to zero. The 
        csvTitles is a list of titles for the csvRow list.
        """
        if len(csvTitles) != len(csvRow): 
            raise ValueError("Length of titles list is different to that of csvRow")
        
        numFields = len(questionIds)
        egoRow = numpy.zeros(numFields) 

        for i in range(0, numFields): 
            try: 
                fieldIndex = csvTitles.index(questionIds[i][0])
            except: 
                logging.debug(("Field not found: " + questionIds[i][0]))
                raise 
            
            if questionIds[i][1] == 0:
                try: 
                    egoRow[i] = float(csvRow[fieldIndex])
                except: 
                    print(("Field has missing values: " + questionIds[i][0]))
                    raise 
            elif questionIds[i][1] == 1:
                egoRow[i] = self.__markMissingValues(csvRow[fieldIndex], 0)
            #This is a missing value we do not want replaced with mean or mode
            #e.g. with alters. 
            elif questionIds[i][1] == 2: 
                egoRow[i] = self.__markMissingValues(csvRow[fieldIndex], -1)
            else:
                raise ValueError("Problem with questionIds field: " + str(questionIds[i][0]))
            
        return egoRow
        
    #This is going to be really slow! 
    def generateNonReceivers(self, egoArray, numContactsIndices, homophileIndexPairs, receiverCounts):
        """
        Generate a series of non receivers from egoArray based on homophility information. 
        egoArray is the array of all egos
        Inputs
        ------
        numContactsIndices - a list of indices of the number of various contacts (friends, family etc.)
        homophileIndexPairs - a list of pairs. The first is the index of the homophility and the second
        is the index of the variable.
        receiverCounts - the number of receivers for each ego 
        Outputs
        -------
        contactsArray - the array of non-receivers
        egoIndices - the corresponding 1D array of ego indices
        """ 
        (numEgos, numEgoFeatures) = (egoArray.shape[0], egoArray.shape[1])
        egoIndices = numpy.zeros(0, numpy.int32) #Store the index of each ego to each contact
        alterIndices = numpy.zeros(0, numpy.int32) #Store the index of each alter to each contact
        contactsArray = numpy.zeros((0, numEgoFeatures))
        
        logging.info("Generating non-receivers for " + str(numEgos) + " egos")
        
        #Assume number of contacts above 9 is just 10 (final category)
        for i in range(0, numEgos):  
            Util.printIteration(i, self.printIterationStep, numEgos)
            
            totalContacts = 0
            for j in range(0, len(numContactsIndices)): 
                totalContacts = totalContacts + int(egoArray[i, numContactsIndices[j]]*2)

            totalContacts = max(0, totalContacts - receiverCounts[i])

            #Get a sample of indices for similar people (remove the current person)
            homophileIndices = numpy.array([], numpy.int)
            
            for j in range(0, len(homophileIndexPairs)): 
                if egoArray[i, homophileIndexPairs[j][0]] == 1:
                    if j==0:
                        homophileIndices = numpy.setdiff1d(numpy.array(list(range(0, numEgos))), numpy.array([i]))

                    subset = numpy.nonzero(egoArray[:, homophileIndexPairs[j][1]] == egoArray[i, homophileIndexPairs[j][1]])[0]
                    homophileIndices = numpy.intersect1d(homophileIndices, subset)
                    
            nonHomophileIndices = numpy.setdiff1d(numpy.array(list(range(0, numEgos))), numpy.array([i]))
            nonHomophileIndices = numpy.setdiff1d(nonHomophileIndices, homophileIndices)

            numHomophileContacts = min(int(round(self.p * totalContacts)), homophileIndices.shape[0])
            numNonHomophilesContacts = min(totalContacts-numHomophileContacts, nonHomophileIndices.shape[0])
            tempContacts =  numpy.zeros((numHomophileContacts+numNonHomophilesContacts, numEgoFeatures))
            
            #Add homophiles 
            perm = Util.sampleWithoutReplacement(numHomophileContacts, homophileIndices.shape[0])
            tempContacts[0:numHomophileContacts, :] = egoArray[homophileIndices[perm], :]
            alterIndices = numpy.r_[alterIndices, homophileIndices[perm]]
                
            #Add non homophiles
            perm = Util.sampleWithoutReplacement(numNonHomophilesContacts, nonHomophileIndices.shape[0])
            tempContacts[numHomophileContacts:numHomophileContacts+numNonHomophilesContacts, :] = egoArray[nonHomophileIndices[perm], :]
            alterIndices = numpy.r_[alterIndices, nonHomophileIndices[perm]]

            tempEgoIndices = numpy.ones(numHomophileContacts+numNonHomophilesContacts) * i 
            
            contactsArray = numpy.r_[contactsArray, tempContacts]
            egoIndices = numpy.r_[egoIndices, tempEgoIndices]
            
        
        logging.info("Done - generated " + str(egoIndices.shape[0]) + " non-receivers")
        return contactsArray, egoIndices, alterIndices
                     
    #We have the assumption that all missing values are zero and all non-missing values are non-zero
    def replaceMissingValues(self, X):
        numCols = X.shape[1]
        
        for i in range(0, numCols): 
            sumCol = sum(X[:, i])
            numNonMissing = sum(X[:, i]!=0) 
            
            if numNonMissing != 0: 
                missingVal = float(sumCol)/numNonMissing
            else:
                missingVal= 0 
                
            X[X[:,i] == 0, i] = missingVal
            
        return X

    #Replace missing values with a random normal variable based on the non-missing values
    def replaceMissingValues2(self, X):
        numCols = X.shape[1]

        for i in range(0, numCols):
            mu = numpy.mean(X[X[:,i] != 0, i])
            var = numpy.var(X[X[:,i] != 0, i])
            numMissing = sum(X[:, i]==0)

            newVals = numpy.random.randn(numMissing)*var+mu
            X[X[:,i] == 0, i] = newVals
            
        return X
       
    #Replace missing values with the mode value
    def replaceMissingValuesMode(self, X):
        numCols = X.shape[1]

        for i in range(0, numCols):
            mode = Util.mode(X[X[:,i] != 0, i])

            X[X[:,i] == 0, i] = mode

        return X

    #Take a string which represents an integer or missing value (empty string).
    def __markMissingValues(self, valString, mark):
        if valString == "" or valString == " ": 
            return mark 
        else:
            return float(valString)
    
        