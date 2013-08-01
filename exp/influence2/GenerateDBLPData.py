
from exp.influence2.DBLPDataset import DBLPDataset
import logging 
import sys 

"""
Create some graphs from the DBLP data. Basically, we use a seed list of experts 
and then find all the coauthors and their publications. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#field = "Boosting" 
field  = "IntelligentAgents"
dataset = DBLPDataset(field)


dataset.splitExperts()
dataset.writeCoauthors()
dataset.writePublications()

