import numpy 
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from apgl.util.PathDefaults import PathDefaults 
from exp.util.IdIndexer import IdIndexer 
import xml.etree.ElementTree as ET
import array 

metadataDir = PathDefaults.getDataDir() + "aps/aps-dataset-metadata-2010/"
metadataFilename = metadataDir + "PRSTAB.xml"

citationsDir = PathDefaults.getDataDir() + "aps/aps-dataset-citations-2010/"
citatonsFilename = citationsDir + "citing_cited.csv"

tree = ET.parse(metadataFilename)
root = tree.getroot()

authorIndexer = IdIndexer("i")
articleIndexer = IdIndexer("i")

for child in root: 
    authorGroups = child.findall('authgrp')    
    
    for authorGroup in authorGroups: 
        authors = authorGroup.findall("author")        
        
        for author in authors: 
            if author.find("givenname") != None: 
                fullname = author.find("givenname").text
            else: 
                fullname = ""
            
            for middlename in author.findall("middlename"): 
                fullname += " " + middlename.text
                
            fullname += " " + author.find("surname").text
            
            authorId = fullname
            articleId = child.attrib["doi"]
           
            authorIndexer.append(authorId)           
            articleIndexer.append(articleId)
            
authorInds = authorIndexer.getArray()
articleInds = articleIndexer.getArray()

#We now need to read the citations file and add those edges 
article1Inds = array.array("i") 
article2Inds = array.array("i")

citationsFile = open(citatonsFilename)
citationsFile.readline()

for line in citationsFile: 
    vals = line.split(",")
    articleId1 = vals[0].strip()
    articleId2 = vals[1].strip()
    
    #print(articleId1, articleId2)
    
    articleIdDict = articleIndexer.getIdDict()
    
    if articleId1 in articleIdDict and articleId2 in articleIdDict: 
        article1Inds.append(articleIdDict[articleId1])
        article2Inds.append(articleIdDict[articleId2])

article1Inds = numpy.array(article1Inds)
article2Inds = numpy.array(article2Inds)

authorArticleEdges = numpy.c_[authorInds, articleInds]
print(authorArticleEdges)

articleArticleEdges = numpy.c_[article1Inds, article2Inds]
print(articleArticleEdges)

print(articleArticleEdges.shape)
      
graph = igraph.Graph()
graph.add_vertices(numpy.max(authorInds) + numpy.max(articleInds))
graph.add_edges(authorArticleEdges)

print(graph.summary())      
