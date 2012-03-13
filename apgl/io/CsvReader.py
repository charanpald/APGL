
import csv

class CsvReader():
    def __init__(self):
        pass

    def getNumLines(self, fileName):
        try:
            reader = csv.reader(open(fileName, "rU"))
        except IOError:
            raise

        numLines = 0
        for row in reader:
            numLines = numLines + 1

        return numLines
