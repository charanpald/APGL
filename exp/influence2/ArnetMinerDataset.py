import os 
import numpy 
import logging 
import difflib 
import re 
import sklearn.feature_extraction.text as text 
import gc
import scipy.sparse
import igraph
import array
import itertools 
import gensim.matutils
import scipy.io
import psutil 
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
import gensim.similarities
from apgl.util.Util import Util 
from apgl.util.PathDefaults import PathDefaults 
from exp.util.IdIndexer import IdIndexer
from collections import Counter, OrderedDict 
from exp.util.PorterTokeniser import PorterTokeniser

class ArnetMinerDataset(object): 
    """
    We process the ArnetMinerDataset into two graphs - a coauthor and an 
    abstract similarity. The output is two graphs - collaboration and 
    abstract similarity. 
    """    
    def __init__(self, k=500, additionalFields=[], runLSI=True):
        numpy.random.seed(21)
        self.runLSI = runLSI 
        self.dataDir = PathDefaults.getDataDir() + "reputation/" 
        
        self.fields = ["Boosting", "Computer Vision", "Cryptography", "Data Mining"]
        self.fields.extend(["Information Extraction", "Intelligent Agents", "Machine Learning"])
        self.fields.extend(["Natural Language Processing", "Neural Networks", "Ontology Alignment"])
        self.fields.extend(["Planning", "Semantic Web", "Support Vector Machine"])    
        self.fields.extend(additionalFields)        
        
        self.dataFilename = self.dataDir + "DBLP-citation-Feb21.txt" 
        #self.dataFilename = self.dataDir + "DBLP-citation-7000000.txt" 
        #self.dataFilename = self.dataDir + "DBLP-citation-100000.txt"        
        self.outputDir = PathDefaults.getOutputDir() + "reputation/"
        self.dataDir = PathDefaults.getDataDir() + "reputation/"
        
        if runLSI:
            self.methodName = "LSI"
        else: 
            self.methodName = "LDA"
            
        self.authorListFilename = self.outputDir + "authorList" + self.methodName +".pkl"
        self.citationListFilename = self.outputDir + "citationList" + self.methodName +".pkl"
        self.vectoriserFilename = self.outputDir + "vectoriser" + self.methodName +".pkl"   
        self.modelFilename = self.outputDir + "model" + self.methodName + ".pkl"
        self.docTermMatrixFilename = self.outputDir + "termDocMatrix" + self.methodName
        self.indexFilename = self.outputDir + "index" + self.methodName 
        self.coverageFilename = self.outputDir + "meanCoverages" + self.methodName + ".npy"
        self.relevantDocsFilename = self.outputDir + "relevantDocs" + self.methodName + ".npy"
        
        self.stepSize = 1000000    
        self.numLines = 15192085
        self.matchCutoff = 1.0   
        
        #Params for finding relevant authors
        self.gamma = 1.3
        self.maxRelevantAuthors = 500
        self.maxRelevantAuthorsMult = 10
        self.printPossibleMatches = False
        self.gammas = numpy.arange(1.0, 2, 0.1)
        self.minExpertArticles = 2

        #Params for vectoriser 
        self.numFeatures = 500000
        self.binary = True 
        self.sublinearTf = False
        self.minDf = 10**-4 
        self.ngram = 2
        self.minDfs = [10**-3, 10**-4, 10**-5]
        
        logging.debug("Limiting BOW/TFIDF features to " + str(self.numFeatures))
        
        #params for LSI        
        self.k = k
        self.q = 3
        self.p = 30 
        self.ks = [100, 200, 300, 400, 500, 600]
        self.sampleDocs = 1000000
        
        self.overwriteGraph = True
        self.overwriteVectoriser = True
        self.overwriteModel = True
        
        self.chunksize = 2000
        self.tfidf = runLSI 
        
        #Load the complete set of experts 
        self.expertsDict = {} 
        for field in self.fields: 
            expertsFile = open(self.getDataFieldDir(field) + "matched_experts.txt")
            self.expertsDict[field] = expertsFile.readlines()
            self.expertsDict[field] = set([x.strip() for x in self.expertsDict[field]])
            expertsFile.close()
            
        #Create a set of experts we use for training 
        self.trainExpertDict = {}
        self.testExpertDict = {}        
        
        for field in self.fields: 
            inds  = numpy.random.permutation(len(self.expertsDict[field]))            
            
            numTrainInds = int(0.5*len(self.expertsDict[field])) 
            trainInds = inds[0:numTrainInds]
            self.trainExpertDict[field] = list(numpy.array(list(self.expertsDict[field]))[trainInds]) 
            
            testInds = inds[numTrainInds:]
            self.testExpertDict[field] = list(numpy.array(list(self.expertsDict[field]))[testInds]) 
            
    def getOutputFieldDir(self, field):
        return self.outputDir + field.replace(' ', '') + "/"
   
    def getDataFieldDir(self, field):
        return self.dataDir + field.replace(' ', '') + "/"
     
    def getCoauthorsFilename(self, field): 
        if self.runLSI: 
            return self.getOutputFieldDir(field) + "coauthorsLSI.pkl"
        else: 
            return self.getOutputFieldDir(field) + "coauthorsLDA.pkl"
        
    def matchExperts(self, relevantExperts, expertsSet): 
        """
        Match experts in the set of relevant experts for the given field. It returns 
        expertMatches which is the name of the expert in the database, and expertSet 
        which is the complete set of experts. 
        """
        expertsSet = set(expertsSet)
        expertMatches = set([])
               
        for relevantExpert in relevantExperts: 
            possibleMatches = difflib.get_close_matches(relevantExpert, expertsSet, cutoff=self.matchCutoff)
            if len(possibleMatches) != 0: 
                expertMatches.add(relevantExpert)
        
        if self.printPossibleMatches: 
            for author in expertsSet.difference(expertMatches):
                possibleMatches = difflib.get_close_matches(author, relevantExperts, cutoff=0.7)
                if len(possibleMatches) != 0: 
                    logging.debug("Possible matches for " + author + " : " + str(possibleMatches))
                          
        expertMatches = sorted(list(expertMatches))
        logging.debug("Total number of matches " + str(len(expertMatches)) + " of " + str(len(expertsSet)))
        
        return expertMatches 

    def coauthorsGraphFromAuthors(self, relevantExperts): 
        """
        Take a set of relevant authors and return the graph. 
        """
        dataFile = open(self.dataFilename)  
        authorIndexer = IdIndexer()
        author1Inds = array.array("i")
        author2Inds = array.array("i")
        
        for relevantExpert in relevantExperts: 
            authorIndexer.append(relevantExpert)
        
        for i, line in enumerate(dataFile):
            Util.printIteration(i, self.stepSize, self.numLines)
            authors = re.findall("#@(.*)", line)  
                            
            if len(authors) != 0: 
                authors = set([x.strip() for x in authors[0].split(",")]) 
                if len(authors.intersection(relevantExperts)) != 0: 
                    iterator = itertools.combinations(authors, 2)
                
                    for author1, author2 in iterator: 
                        if author1 in relevantExperts and author2 in relevantExperts: 
                            author1Ind = authorIndexer.append(author1) 
                            author2Ind = authorIndexer.append(author2)
                                
                            author1Inds.append(author1Ind)
                            author2Inds.append(author2Ind)
        
        logging.debug("Found " + str(len(authorIndexer.getIdDict())) + " coauthors")
                               
        #Coauthor graph is undirected 
        author1Inds = numpy.array(author1Inds, numpy.int)
        author2Inds = numpy.array(author2Inds, numpy.int)
        edges = numpy.c_[author1Inds, author2Inds]            
        
        graph = igraph.Graph()
        graph.add_vertices(len(authorIndexer.getIdDict()))
        graph.add_edges(edges)
        graph.es["weight"] = numpy.ones(graph.ecount())
        graph.simplify(combine_edges=sum)   
        graph.es["invWeight"] = 1.0/(numpy.array(graph.es["weight"])) 
        
        return graph, authorIndexer
            
    def coauthorsGraph(self, field, relevantExperts): 
        """
        Using the relevant authors we find all coauthors. 
        """  
        if not os.path.exists(self.getCoauthorsFilename(field)) or self.overwriteGraph: 
            logging.debug("Finding coauthors of relevant experts")
            graph, authorIndexer = self.coauthorsGraphFromAuthors(set(relevantExperts))
            logging.debug(graph.summary())
            if not os.path.exists(self.getCoauthorsFilename(field)):
                os.makedirs(os.path.dirname(self.getCoauthorsFilename(field)))
            Util.savePickle([graph, authorIndexer], self.getCoauthorsFilename(field), debug=True)
        else: 
            logging.debug("Files already generated: " + self.getCoauthorsFilename(field))  

        graph, authorIndexer = Util.loadPickle(self.getCoauthorsFilename(field))
        return graph, authorIndexer 

    def expertsFromDocSimilarities(self, similarities, numTrainExperts, field): 
        """
        Given a set of similarities work out which documents are relevent 
        and then return a list of ranked authors using these scores. 
        """
        similarities2 = similarities + 1 
        relevantDocs = numpy.arange(similarities2.shape[0])[similarities2 >= self.gamma]
        
        #Save relevant docs 
        numpy.save(self.getOutputFieldDir(field) + "relevantDocs" + self.methodName +  ".npy", relevantDocs)
        
        maxRelevantAuthors = numTrainExperts * self.maxRelevantAuthorsMult  
        
        #Now find all authors corresponding to the documents 
        expertDict = {} 
        expertDict2 = {} 
        expertCitations = {}
        expertsSet = set([])
        
        for docInd in relevantDocs: 
            for author in self.authorList[docInd]: 
                if author not in expertsSet: 
                    expertsSet.add(author)
                    expertDict[author] = similarities2[docInd]
                    expertDict2[author] = 1
                    expertCitations[author] = self.citationList[docInd]
                else: 
                    expertDict[author] += similarities2[docInd] 
                    expertDict2[author] += 1
                    expertCitations[author] += self.citationList[docInd]
        
        expertDict = OrderedDict(sorted(expertDict.items(), key=lambda t: t[1], reverse=True))
        expertsByDocSimilarity = expertDict.keys()[0:maxRelevantAuthors]
        
        expertCitations = OrderedDict(sorted(expertCitations.items(), key=lambda t: t[1], reverse=True))
        expertsByCitations = expertCitations.keys()[0:maxRelevantAuthors]        
        
        logging.debug("Found " + str(len(expertsByDocSimilarity)) + " relevant authors")        
        
        #Remove experts who have written fewer than x articles on the query subject 
        newExpertsByDocSimilarity = []
        for expert in expertsByDocSimilarity: 
            if expertDict2[expert] >= self.minExpertArticles: 
                newExpertsByDocSimilarity.append(expert)
                
        newExpertsByCitations = []
        for expert in expertsByCitations: 
            if expertDict2[expert] >= self.minExpertArticles: 
                newExpertsByCitations.append(expert)
        
        logging.debug("Found " + str(len(newExpertsByDocSimilarity)) + " relevant authors by similarity after culling")
        logging.debug("Found " + str(len(newExpertsByCitations)) + " relevant authors by citation after culling")
        
        return newExpertsByDocSimilarity, newExpertsByCitations 

    def readAuthorsAndDocuments(self): 
        logging.debug("About to read file " + self.dataFilename)
        inFile = open(self.dataFilename)  
        authorList = []
        citationList = []
        documentList = []
                    
        lastAbstract = ""
        lastVenue = ""
        lastTitle = ""    
        lastAuthors = []     
        lastCitationNo = 0                
                    
        for i, line in enumerate(inFile):
            Util.printIteration(i, self.stepSize, self.numLines)
                
            #Match the fields in the file 
            emptyLine = line == "\n"
            title = re.findall("#\*(.*)", line)
            currentAuthors = re.findall("#@(.*)", line)  
            abstract = re.findall("#!(.*)", line)
            venue = re.findall("#conf(.*)", line)
            citationNo = re.findall("#citation(.*)", line)
            
            if emptyLine:
                document = lastTitle + " " + lastAbstract 
                documentList.append(document) 
                authorList.append(lastAuthors)
                citationList.append(lastCitationNo)

                lastAbstract = ""
                lastTitle = ""
                lastAuthors = []
                lastCitationNo = 0   
 
            if len(title) != 0 and len(title[0]) != 0: 
                lastTitle = title[0]
                
            if len(venue) != 0 and len(venue[0]) != 0: 
                lastVenue = venue[0]  
            
            if len(abstract) != 0 and len(abstract[0]) != 0: 
                lastAbstract = abstract[0]
                
            if len(citationNo) != 0 and len(citationNo[0]) != 0: 
                lastCitationNo = int(citationNo[0])
                       
            if len(currentAuthors) != 0: 
                currentAuthors = currentAuthors[0].split(",")  
                currentAuthors = set([x.strip() for x in currentAuthors])
                currentAuthors = currentAuthors.difference(set([""]))
                lastAuthors = currentAuthors                     

        inFile.close() 
        logging.debug("Finished reading " + str(len(documentList)) + " articles")  
        
        return authorList, documentList, citationList

    def vectoriseDocuments(self):
        """
        We want to go through the dataset and vectorise all the title+abstracts.
        The results are saved in TDIDF format in a matrix X. 
        """
        if not os.path.exists(self.docTermMatrixFilename + ".mtx") or not os.path.exists(self.authorListFilename) or not os.path.exists(self.vectoriserFilename) or self.overwriteVectoriser:
            logging.debug("Vectorising documents")            
            
            authorList, documentList, citationList = self.readAuthorsAndDocuments()
            Util.savePickle(authorList, self.authorListFilename, debug=True)
            Util.savePickle(citationList, self.citationListFilename, debug=True)
            
            #vectoriser = text.HashingVectorizer(ngram_range=(1,2), binary=self.binary, norm="l2", stop_words="english", tokenizer=PorterTokeniser(), dtype=numpy.float)
            
            #if self.tfidf: 
            logging.debug("Generating TFIDF features")
            vectoriser = text.TfidfVectorizer(min_df=self.minDf, ngram_range=(1,self.ngram), binary=self.binary, sublinear_tf=self.sublinearTf, norm="l2", max_df=0.95, stop_words="english", tokenizer=PorterTokeniser(), max_features=self.numFeatures, dtype=numpy.float)
            #else: 
            #    logging.debug("Generating bag of word features")
            #    vectoriser = text.CountVectorizer(min_df=self.minDf, ngram_range=(1,self.ngram), binary=False, max_df=0.95, stop_words="english", max_features=self.numFeatures, dtype=numpy.float, tokenizer=PorterTokeniser())            
            
            X = vectoriser.fit_transform(documentList)
            del documentList
            scipy.io.mmwrite(self.docTermMatrixFilename, X)
            logging.debug("Wrote X with shape " + str(X.shape) + " and " + str(X.nnz) + " nonzeros to file " + self.docTermMatrixFilename + ".mtx")
            del X 
                
            #Save vectoriser - note that we can't pickle the tokeniser so it needs to be reset when loaded 
            vectoriser.tokenizer = None 
            Util.savePickle(vectoriser, self.vectoriserFilename, debug=True) 
            del vectoriser  
            gc.collect()
        else: 
            logging.debug("Author list, document-term matrix and vectoriser already generated: ")   
   
    def loadVectoriser(self): 
        self.vectoriser = Util.loadPickle(self.vectoriserFilename)
        self.vectoriser.tokenizer = PorterTokeniser() 
        self.authorList = Util.loadPickle(self.authorListFilename)     
        self.citationList = Util.loadPickle(self.citationListFilename)  
        logging.debug("Loaded vectoriser, citation and author list")
  
    def modelSelection(self): 
        if self.runLSI: 
            meanCoverages = self.modelSelectionLSI()
        else: 
            meanCoverages = self.modelSelectionLDA()
            
        numpy.save(self.coverageFilename, meanCoverages)
        logging.debug("Saved coverages as " + self.coverageFilename)
   
    def learnModel(self):
        if self.runLSI: 
            self.computeLSI()
        else: 
            self.computeLDA()
            
    def findSimilarDocuments(self, field): 
        if self.runLSI: 
            return self.findSimilarDocumentsLSI(field)
        else: 
            return self.findSimilarDocumentsLDA(field)        
      
    def computeLDA(self):
        if not os.path.exists(self.modelFilename) or self.overwriteModel:
            self.vectoriseDocuments()
            self.loadVectoriser()
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.docTermMatrixFilename + ".mtx")
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))   
            
            logging.getLogger('gensim').setLevel(logging.INFO)
            lda = LdaModel(corpus, num_topics=self.k, id2word=id2WordDict, chunksize=self.chunksize, distributed=False) 
            #index = gensim.similarities.docsim.SparseMatrixSimilarity(lda[corpus], num_features=self.k) 
            index = gensim.similarities.docsim.Similarity(self.indexFilename, lda[corpus], num_features=self.k)            
            
            Util.savePickle([lda, index], self.modelFilename, debug=True)
            gc.collect()
        else: 
            logging.debug("File already exists: " + self.modelFilename)
   
    def findSimilarDocumentsLDA(self, field): 
        """ 
        We use LDA in this case 
        """
        self.computeLDA()
        self.loadVectoriser()
                            
        lda, index = Util.loadPickle(self.modelFilename)
        
        newX = self.vectoriser.transform([field])
        newX = [(i, newX[0, i])for i in newX.nonzero()[1]]
        result = lda[newX]    
        
        #Cosine similarity 
        similarities = index[result]
        expertsByDocSimilarity, expertsByCitations = self.expertsFromDocSimilarities(similarities, len(self.trainExpertDict[field]), field)
        
        logging.debug("Number of relevant authors : " + str(len(expertsByDocSimilarity)))
        return expertsByDocSimilarity, expertsByCitations

    def modelSelectionLDA(self): 
        """
        Lets find the optimal parameters for LDA for all fields. We see the optimal 
        number of parameters for the training set of experts. 
        """
        coverages = numpy.zeros((len(self.ks), len(self.minDfs), len(self.gammas), len(self.fields)))
        logging.getLogger('gensim').setLevel(logging.INFO) 
        
        logging.debug("Starting model selection for LDA")       
       
        for t, minDf in enumerate(self.minDfs): 
            logging.debug("Using minDf=" + str(minDf))
            self.minDf = minDf
            
            self.vectoriseDocuments()
            self.loadVectoriser()
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.docTermMatrixFilename + ".mtx")
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))
                            
            for i, k in enumerate(self.ks): 
                logging.debug("Running LDA with " + str(k) + " dimensions")
                lda = LdaModel(corpus, num_topics=k, id2word=id2WordDict, chunksize=self.chunksize, distributed=False)  
                logging.debug("Creating index")
    
                index = gensim.similarities.docsim.Similarity(self.indexFilename, lda[corpus], num_features=k)
                
                for j, field in enumerate(self.fields): 
                    logging.debug("k="+str(k) + " and field=" + str(field))                
                    newX = self.vectoriser.transform([field])
                    newX = [(s, newX[0, s])for s in newX.nonzero()[1]]
                    result = lda[newX]             
                    similarities = index[result]
                    
                    for u, gamma in enumerate(self.gammas): 
                        self.gamma = gamma 
                        expertsByDocSimilarity, expertsByCitations = self.expertsFromDocSimilarities(similarities, len(self.trainExpertDict[field]), field)
                        
                        expertMatches = self.matchExperts(expertsByDocSimilarity, set(self.trainExpertDict[field]))
                        coverages[i, t, u, j] = float(len(expertMatches))/len(self.trainExpertDict[field])
                
                for u, gamma in enumerate(self.gammas):
                    logging.debug("Mean coverage for gamma=" + str(gamma) + " " + str(numpy.mean(coverages[i, t, u, :])))
            
        meanCoverges = numpy.mean(coverages, 3)
        logging.debug(meanCoverges)
        
        bestInds = numpy.unravel_index(numpy.argmax(meanCoverges), meanCoverges.shape)           
        
        self.k = self.ks[bestInds[0]]
        logging.debug("Chosen k=" + str(self.k))
        
        self.minDf = self.minDfs[bestInds[1]]
        logging.debug("Chosen minDf=" + str(self.minDf))
        
        self.gamma = self.gammas[bestInds[2]]
        logging.debug("Chosen gamma=" + str(self.gamma))  
        
        logging.debug("Coverage = " + str(numpy.max(meanCoverges)))
                
        return meanCoverges

    def computeLSI(self):
        """
        Compute using the LSI version in gensim 
        """
        if not os.path.exists(self.modelFilename) or self.overwriteModel:
            self.vectoriseDocuments()
            self.loadVectoriser()
            #X = scipy.io.mmread(self.docTermMatrixFilename)
            #corpus = gensim.matutils.MmReader(self.docTermMatrixFilename + ".mtx", True)
            #corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.docTermMatrixFilename + ".mtx")
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))   
            
            logging.getLogger('gensim').setLevel(logging.INFO)
            lsi = LsiModel(corpus, num_topics=self.k, id2word=id2WordDict, chunksize=self.chunksize, distributed=False) 
            index = gensim.similarities.docsim.Similarity(self.indexFilename, lsi[corpus], num_features=self.k)          
            
            Util.savePickle([lsi, index], self.modelFilename, debug=True)
            gc.collect()
        else: 
            logging.debug("File already exists: " + self.modelFilename)   

    def findSimilarDocumentsLSI(self, field): 
        """ 
        We use LSI from gensim in this case 
        """
        self.computeLSI()
        self.loadVectoriser()
                            
        lsi, index = Util.loadPickle(self.modelFilename)
        newX = self.vectoriser.transform([field])
        newX = [(i, newX[0, i])for i in newX.nonzero()[1]]
        result = lsi[newX]             
        similarities = index[result]
        expertsByDocSimilarity, expertsByCitations = self.expertsFromDocSimilarities(similarities, len(self.trainExpertDict[field]), field)
        
        return expertsByDocSimilarity, expertsByCitations

    def modelSelectionLSI(self): 
        """
        Lets find the optimal parameters for LSI for all fields. We see the optimal 
        number of parameters for the training set of experts. 
        """
       
        coverages = numpy.zeros((len(self.ks), len(self.minDfs), len(self.gammas), len(self.fields)))
        logging.getLogger('gensim').setLevel(logging.INFO) 
        maxK = numpy.max(self.ks)
        
        logging.debug("Starting model selection for LSI")       
       
        for t, minDf in enumerate(self.minDfs): 
            logging.debug("Using minDf=" + str(minDf))
            self.minDf = minDf
            
            self.vectoriseDocuments()
            self.loadVectoriser()
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.docTermMatrixFilename + ".mtx")
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))
            
            logging.debug("Running LSI with " + str(maxK) + " dimensions")
            lsi = LsiModel(corpus, num_topics=maxK, id2word=id2WordDict, chunksize=self.chunksize, distributed=False, onepass=False)    
            
            for i, k in enumerate(self.ks): 
                lsi.num_topics = k
                logging.debug("Creating index")
                index = gensim.similarities.docsim.Similarity(self.indexFilename, lsi[corpus], num_features=k)
                
                for j, field in enumerate(self.fields): 
                    logging.debug("k="+str(k) + " and field=" + str(field))                
                    newX = self.vectoriser.transform([field])
                    newX = [(s, newX[0, s])for s in newX.nonzero()[1]]
                    result = lsi[newX]             
                    similarities = index[result]
                    
                    for u, gamma in enumerate(self.gammas): 
                        self.gamma = gamma 
                        expertsByDocSimilarity, expertsByCitations = self.expertsFromDocSimilarities(similarities, len(self.trainExpertDict[field]), field)
                        
                        expertMatches = self.matchExperts(expertsByDocSimilarity, set(self.trainExpertDict[field]))
                        coverages[i, t, u, j] = float(len(expertMatches))/len(self.trainExpertDict[field])
                
                for u, gamma in enumerate(self.gammas):
                    logging.debug("Mean coverage for gamma=" + str(gamma) + " " + str(numpy.mean(coverages[i, t, u, :])))
            
        meanCoverges = numpy.mean(coverages, 3)
        logging.debug(meanCoverges)
        
        bestInds = numpy.unravel_index(numpy.argmax(meanCoverges), meanCoverges.shape)           
        
        self.k = self.ks[bestInds[0]]
        logging.debug("Chosen k=" + str(self.k))
        
        self.minDf = self.minDfs[bestInds[1]]
        logging.debug("Chosen minDf=" + str(self.minDf))
        
        self.gamma = self.gammas[bestInds[2]]
        logging.debug("Chosen gamma=" + str(self.gamma))   
        
        logging.debug("Coverage = " + str(numpy.max(meanCoverges)))
        
        return meanCoverges
        
        