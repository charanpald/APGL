from lxml import etree
import HTMLParser
import logging 
import difflib 
import re
import os 
import gc 
import numpy 
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 

class DBLPDataset(object): 
    def __init__(self, field):
        numpy.random.seed(21)        
        
        dataDir = PathDefaults.getDataDir() + "dblp/"
        self.xmlFileName = dataDir + "dblp.xml"
        self.xmlCleanFilename = dataDir + "dblpClean.xml"        

        resultsDir = PathDefaults.getDataDir() + "reputation/" + field + "/"
        self.expertsFileName = resultsDir + "experts.txt"
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.trainExpertMatchesFilename = resultsDir + "experts_train_matches.csv"
        self.testExpertMatchesFilename = resultsDir + "experts_test_matches.csv"
        self.coauthorsFilename = resultsDir + "coauthors.csv"
        self.publicationsFilename = resultsDir + "publications.csv"
        
        self.stepSize = 100000
        self.numLines = 33532888
        self.publicationTypes = set(["article" , "inproceedings", "proceedings", "book", "incollection", "phdthesis", "mastersthesis", "www"])
        self.p = 0.5     
        
        
        self.cleanXML()
        self.matchExperts()
        logging.warning("Now you must disambiguate the matched experts if not ready done")        
        
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
            
    def loadExperts(self, fileName): 
        #Load the experts 
        seedFile = open(fileName)
        expertsSet = set([])
        
        for line in seedFile: 
            expertsSet.add(line.strip())
            
        seedFile.close()         
        return expertsSet 
    
    def cleanXML(self):
        """
        Take the original XML file and clean up HTML characters and & symbols. We 
        also create a list of possible matches for the experts. 
        """
        if not os.path.exists(self.xmlCleanFilename):
            logging.debug("Cleaning XML")
            h = HTMLParser.HTMLParser()
            
            inFile = open(self.xmlFileName)
            outFile = open(self.xmlCleanFilename, "w")
            i = 0 
            
            for line in inFile: 
                Util.printIteration(i, self.stepSize, self.numLines)
                outLine = h.unescape(line).replace("&", "&amp;")
                outLine = re.sub("<title>.*[\<\>].*</title>", "<title>Default Title</title>", outLine)
                outLine = re.sub("<ee>.*[\<\>].*</ee>", "<ee>Default text</ee>", outLine)
                outFile.write(outLine) 
                i += 1
            
            inFile.close() 
            outFile.close() 
            logging.debug("All done")
        else: 
            logging.debug("File already generated: " + self.xmlCleanFilename)
    
    def matchExperts(self): 
        expertsSet = self.loadExperts(self.expertsFileName)
        
        if not os.path.exists(self.expertMatchesFilename): 
            inFile = open(self.xmlCleanFilename)    
            expertMatches = set([])
            i = 0 
            
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                if i % self.stepSize == 0: 
                    logging.debug(expertMatches)
                    
                author = re.findall("<author>(.*)</author>", line)  
                if len(author) != 0: 
                    possibleMatches = difflib.get_close_matches(author[0], expertsSet, cutoff=0.9)
                    if len(possibleMatches) != 0: 
                        expertMatches.add(author[0]) 
                
                i += 1
            
            expertMatches = sorted(list(expertMatches))
            expertMatchesFile = open(self.expertMatchesFilename, "w")
            
            for expert in expertMatches: 
                expertMatchesFile.write(expert + "\n")
            expertMatchesFile.close()
            
            logging.debug("All done")
        else: 
            logging.debug("File already generated: " + self.expertMatchesFilename)
        
        
    def writeCoauthors(self): 
        if not os.path.exists(self.coauthorsFilename): 
            matchedExperts = self.loadMatchedExperts()
                
            expertCoauthors = set([])
            xml = open(self.xmlCleanFilename)
            publicationTypes = set(["article" , "inproceedings", "proceedings", "book", "incollection", "phdthesis", "mastersthesis", "www"])
            iterator = etree.iterparse(xml, events=('start', 'end'))
            i = 0        
            
            while True: 
                try: 
                    event, elem = iterator.next()     
                    if elem.tag in publicationTypes and event=="start": 
                        articleDict = {}
                        articleDict["authors"] = set([])
                        articleDict["field"] = 0 
                        articleDict["id"] = elem.get("key")
                    
                    if elem.tag == "author" and event=="start" and elem.text != None:
                        articleDict["authors"].add(elem.text)
        
                    if elem.tag == "year" and event=="start":
                        articleDict["year"] = elem.text
                    
                    if elem.tag in publicationTypes and event=="end": 
                        for author in articleDict["authors"]: 
                            #print(articleDict["id"], author, articleDict["year"])
                            if author in matchedExperts:      
                                expertCoauthors = expertCoauthors.union(articleDict["authors"])
                                logging.debug("Added coauthor(s) of " + author + ": " + str(articleDict["authors"]))
                        elem.clear()
                    
                    if i % 100000 == 0: 
                        gc.collect()
                    i += 1
                except etree.XMLSyntaxError as err: 
                    print(err) 
                    raise 
                except StopIteration: 
                    break 
                            
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
        
    def writePublications(self): 
        if not os.path.exists(self.publicationsFilename): 
            #Read the set of matched experts 
            coauthors = set([])        
            coauthorsFile = open(self.coauthorsFilename)    
            for line in coauthorsFile: 
                coauthors.add(line.strip())
            coauthorsFile.close()
                
            xml = open(self.xmlCleanFilename)
            publicationsFile = open(self.publicationsFilename, "w")
            
            iterator = etree.iterparse(xml, events=('start', 'end'))
            i = 0        
            
            while True: 
                try: 
                    event, elem = iterator.next()     
                    if elem.tag in self.publicationTypes and event=="start": 
                        articleDict = {}
                        articleDict["authors"] = set([])
                        articleDict["field"] = 0 
                        articleDict["id"] = elem.get("key")
                        articleDict["save"] = False
                        articleDict["year"] = None 
                    
                    if elem.tag == "author" and event=="start" and elem.text != None:
                        articleDict["authors"].add(elem.text)
                        if elem.text in coauthors: 
                            articleDict["save"] = True
        
                    if elem.tag == "year" and event=="start":
                        articleDict["year"] = elem.text
                    
                    if elem.tag in self.publicationTypes and event=="end": 
                        if articleDict["save"]: 
                            for j, author in enumerate(articleDict["authors"]): 
                                #Format: z/Zhang:Cha;journals/tmm/ZhangYRCVSPZ08;3;1;2008
                                outputLine = author + ";" + articleDict["id"] + ";" + str(j) + ";0;" + str(articleDict["year"]) + "\n"
                                publicationsFile.write(outputLine)
    
                        elem.clear()
                    
                    if i % 100000 == 0: 
                        gc.collect()
                        logging.debug("Element events: " + str(i))
                    i += 1
                except etree.XMLSyntaxError as err: 
                    print(err) 
                    raise 
                except StopIteration: 
                    break  
                            
            xml.close() 
            publicationsFile.close()
            logging.debug("Total number of author/paper pairs: " + str(i))
            logging.debug("Wrote publications to file " + str(self.publicationsFilename))
        else: 
            logging.debug("File already generated: " + str(self.publicationsFilename))