from apgl.util.PathDefaults import PathDefaults 
from lxml import etree
import HTMLParser
import os
import logging 
import sys 
import difflib 
import re

"""
Create some graphs from the DBLP data. Basically, we use a seed list of experts 
and then find all the coauthors and their publications. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def removeSpecialChars(): 
    publicationsXML = dataDir + "dblp.xml"
    outFileName = dataDir + "dblp2.xml"
    
    if not os.path.exists(outFileName):
        logging.debug("Cleaning XML")
        h = HTMLParser.HTMLParser()
        
        inFile = open(publicationsXML)
        outFile = open(outFileName, "w")
        
        for line in inFile: 
            outLine = h.unescape(line).replace("&", "&amp;")
            
            outLine = re.sub("<title>.*[\<\>].*</title>", "<title>Default Title</title>", outLine)
            outLine = re.sub("<ee>.*[\<\>].*</ee>", "<ee>Default text</ee>", outLine)
            outFile.write(outLine) 
        
        inFile.close() 
        outFile.close() 
        
        logging.debug("All done")
    else: 
        logging.debug("File already processed: " + publicationsXML)

field = "Boosting" 
dataDir = PathDefaults.getDataDir() + "dblp/"

publicationsXML2 = dataDir + "dblp2.xml"
seedFileName = PathDefaults.getDataDir() + "reputation/" + field + "/" + field.lower() + "_seed_train.csv"

#Load the experts 
seedFile = open(seedFileName)
expertsSet = set([])

for line in seedFile: 
    vals = line.split(",")
    expertsSet.add(vals[1].strip() + " " + vals[0].strip())
    
seedFile.close() 
print(expertsSet)

removeSpecialChars()
publicationTypes = set(["article" , "inproceedings", "proceedings", "book", "incollection", "phdthesis", "mastersthesis", "www"])

#On first pass find the coauthors of the experts 
expertCoauthors = []
xml = open(publicationsXML2)

iterator = etree.iterparse(xml, events=('start', 'end', 'start-ns', "end-ns"))
while True: 
    try: 
        event, elem = iterator.next()
    
        if elem.tag in publicationTypes and event=="start": 
            articleDict = {}
            articleDict["authors"] = []
            articleDict["field"] = 0 
            articleDict["id"] = elem.get("key")
            
        if elem.tag == "author" and event=="start" and elem.text != None:
            articleDict["authors"].append(elem.text)
    
        if elem.tag == "year" and event=="start":
            articleDict["year"] = elem.text
            
        if elem.tag in publicationTypes and event=="end": 
            for author in articleDict["authors"]: 
                #print(articleDict["id"], author, articleDict["year"])
                possibleMatches = difflib.get_close_matches(author, expertsSet, cutoff=0.8) 
                if len(possibleMatches) != 0:      
                    expertCoauthors.append(author)
                    logging.debug("Added coauthor of " + possibleMatches[0] + ": " + author)
    except etree.XMLSyntaxError as e: 
        logging.debug("Error in XML Syntax detected: " + str(e))
        raise 
    except StopIteration: 
        break 
        
logging.debug("Found " + str(len(expertCoauthors)) + " coauthors")