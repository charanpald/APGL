from apgl.util.PathDefaults import PathDefaults 
from lxml import etree
"""
Create some graphs from the DBLP data. Basically, we use a seed list of experts 
and then find all the coauthors and their publications. 
"""

def removeSpecialChars(publicationsXML): 
    import HTMLParser
    h = HTMLParser.HTMLParser()
    
    outFileName = dataDir + "dblp2.xml"
    
    inFile = open(publicationsXML)
    outFile = open(outFileName, "w")
    
    for line in inFile: 
        outFile.write(h.unescape(line).replace("&", "&amp;")) 
    
    inFile.close() 
    outFile.close() 
    
    print("All done")

field = "Boosting" 
dataDir = PathDefaults.getDataDir() + "dblp/"
publicationsXML = dataDir + "dblp.xml"
publicationsXML2 = dataDir + "dblp2.xml"
seedFileName = PathDefaults.getDataDir() + "reputation/" + field + "/"

#removeSpecialChars(publicationsXML)

xml = open(publicationsXML2)
for event, elem in etree.iterparse(xml, events=('start', 'end', 'start-ns', "end-ns")):
    if elem.tag == "author" and event=="start": 
        print(elem.text)

