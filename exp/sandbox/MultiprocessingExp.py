
import sys
import logging
import numpy
import multiprocessing
import time
from datetime import datetime

"""
Some code to test shared variables in multiprocess code 
"""

class RandChooser(multiprocessing.Process):
    def __init__(self, args):
        super(RandChooser, self).__init__(args=args)
        self.args = args 
        
        dt = datetime.now()
        numpy.random.seed(dt.microsecond)


    def chooseRand(self):
        i = numpy.random.rand()
        logging.info("Chose new value:" + str(i))
        logging.info("Min value is " + str(self.args[0].value))

        if i<self.args[0].value:
            self.args[0].value = i
            for j in range(len(self.args[1])-1, 0, -1):
                self.args[1][j] = self.args[1][j-1]

            self.args[1][0] = i

            for j in range(len(self.args[1])):
                print((self.args[1][j]))
            logging.info("New minVal=" + str(i))

    def run(self):
        while(True):
            self.chooseRand()
            time.sleep(2)

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)

minVal = multiprocessing.Value('d', 10)
minList = multiprocessing.Array('d', [0 for i in range(10)])
args = (minVal, minList)

numProcesses = 5
for i in range(numProcesses):
    chooser = RandChooser(args)
    chooser.start()

#Just use lists 