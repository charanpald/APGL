'''
Created on 3 Jul 2009

@author: charanpal

An abstract base class which represents a graph generator. The graph generator
takes an existing empty graph and produces edges over it. 
'''
from apgl.util.Util import Util

class AbstractGraphGenerator(object):
    def generate(self, graph):
        Util.abstract() 