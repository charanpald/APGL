import os 
import numpy 
import logging 
import difflib 
import re 
from apgl.util.Util import Util 
from apgl.util.PathDefaults import PathDefaults 

class ArnetMinerDataset(object): 
    """
    We process the ArnetMinerDataset into two graphs - a coauthor and an 
    abstract similarity. The output is two graphs - collaboration and 
    abstract similarity. 
    """    
    def __init__(self, field):
        numpy.random.seed(21)
        dataDir = PathDefaults.getDataDir() + "dblpCitation/" 
        self.dataFilename = dataDir + "DBLP-citation-Feb21.txt" 
        
        resultsDir = PathDefaults.getDataDir() + "reputation/" + field + "/"
        self.expertsFileName = resultsDir + "experts.txt"
        
        resultsDir += "arnetminer/"
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.trainExpertMatchesFilename = resultsDir + "experts_train_matches.csv"
        self.testExpertMatchesFilename = resultsDir + "experts_test_matches.csv"
        self.coauthorsFilename = resultsDir + "coauthors.csv"
        self.coauthorSimilarityFilename = resultsDir + "coauthorSimilarity.csv"        
        
        self.stepSize = 100000    
        self.numLines = 15192085
        self.matchCutoff = 0.95
        self.p = 0.5
        
        self.matchExperts()
        self.splitExperts()   
        self.writeCoauthors()
        self.writeSimilarityGraph()
        
    def matchExperts(self): 
        expertsFile = open(self.expertsFileName)
        expertsSet = expertsFile.readlines()
        expertsSet = set([x.strip() for x in expertsSet])
        
        if not os.path.exists(self.expertMatchesFilename): 
            inFile = open(self.dataFilename)    
            expertMatches = set([])
            i = 0 
            
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                if i % self.stepSize == 0: 
                    logging.debug(expertMatches)
                    
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = authors[0].split(",")                        
                    for author in authors: 
                        possibleMatches = difflib.get_close_matches(author, expertsSet, cutoff=self.matchCutoff)
                        if len(possibleMatches) != 0: 
                            expertMatches.add(author)
                            expertsSet.remove(possibleMatches[0])
                            
                            if len(expertsSet) == 0: 
                                logging.debug("Found all experts, breaking")
                                break  
                i += 1
            
            expertMatches = sorted(list(expertMatches))
            expertMatchesFile = open(self.expertMatchesFilename, "w")
            
            for expert in expertMatches: 
                expertMatchesFile.write(expert + "\n")
            expertMatchesFile.close()
            
            logging.debug("All done")
        else: 
            logging.debug("File already generated: " + self.expertMatchesFilename)
        
    def splitExperts(self): 
        if not (os.path.exists(self.trainExpertMatchesFilename) and os.path.exists(self.testExpertMatchesFilename)): 
            file = open(self.expertMatchesFilename) 
            logging.debug("Splitting list of experts given in file " +  self.expertMatchesFilename)
            trainNames = [] 
            testNames = []
            
            for line in file: 
                if numpy.random.rand() < self.p: 
                    trainNames.append(line.strip() + "\n")
                else: 
                    testNames.append(line.strip() + "\n")  
                    
            file.close() 
            
            logging.debug("Train names: " + str(len(trainNames)) + " test names: " + str(len(testNames))) 
            
            trainFile = open(self.trainExpertMatchesFilename, "w") 
            trainFile.writelines(trainNames) 
            trainFile.close() 
            
            testFile = open(self.testExpertMatchesFilename, "w") 
            testFile.writelines(testNames) 
            testFile.close() 
            
            logging.debug("Wrote train and test names to " +  self.trainExpertMatchesFilename + " and " + self.testExpertMatchesFilename) 
        else: 
            logging.debug("Generated files exist: " + self.trainExpertMatchesFilename + " " + self.testExpertMatchesFilename)
            
    def writeCoauthors(self): 
        if not os.path.exists(self.coauthorsFilename): 
            matchedExpertsFile = open(self.trainExpertMatchesFilename)
            matchedExperts = matchedExpertsFile.readlines()
            matchedExperts = set([x.strip() for x in matchedExperts])
            
            
            inFile = open(self.dataFilename)
            i = 0     
            
            expertCoauthors = set([])
            
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = authors[0].split(",")  
                    authors = [x.strip() for x in authors]     
                    for author in authors: 
                        if author in matchedExperts: 
                            expertCoauthors = expertCoauthors.union(set(authors))
                            
                i += 1
            
            logging.debug("Found " + str(len(expertCoauthors)) + " coauthors")
            
            #Now just write out the coauthors 
            coauthorsFile = open(self.coauthorsFilename, "w")
            expertCoauthors = sorted(list(expertCoauthors))
            
            for coauthor in expertCoauthors: 
                coauthorsFile.write(coauthor + "\n")
            coauthorsFile.close()
            
            logging.debug("Wrote coauthors to file " + str(self.coauthorsFilename))
        else: 
            logging.debug("File already generated: " + self.coauthorsFilename)  
    
    def writeSimilarityGraph(self): 
        """
        Run through the abstracts and compare them from each author. 
        """
        abstractDict = {} 
        
        coauthorsFile = open(self.coauthorsFilename)
        coauthors = coauthorsFile.readlines()
        coauthors = set([x.strip() for x in coauthors])
        
        for coauthor in coauthors: 
            abstractDict[coauthor] = ""    
        
        if not os.path.exists(self.coauthorSimilarityFilename): 
            inFile = open(self.dataFilename)
            i = 0     
                        
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)

                if line == "\n": 
                    authors = []
                    
                #Authors come before the abstract 
                currentAuthors = re.findall("#@(.*)", line)  
                abstract = re.findall("#!(.*)", line) 
                
                if len(abstract) != 0 and len(abstract[0]) != 0: 
                    lastAbstract = abstract[0]
                           
                if len(currentAuthors) != 0: 
                    currentAuthors = currentAuthors[0].split(",")  
                    currentAuthors = set([x.strip() for x in currentAuthors])
                    currentAuthors = coauthors.intersection(currentAuthors)                           
                            
                #One of the coauthors wrote this paper 
                if len(currentAuthors) != 0: 
                    for author in currentAuthors: 
                        if author in coauthors: 
                            print(i, author, abstract)
                            abstractDict[author] = abstractDict[author] + " " + lastAbstract
                             
                i += 1
