from datetime import date, timedelta

class DateUtils(object):
    """
    This class stores some useful functions operating with dates
    """
    def __init__(self):
        pass

    @staticmethod
    def getDateStrFromDay(day, startYear):
        """
        Takes a day and start year and add the number of days to the start date
        and return the string representation. 
        """
        startDate = date(startYear, 1, 1)
        tDelta = timedelta(days=day)
        endDate = startDate + tDelta
        return endDate.strftime("%d/%m/%y")

    @staticmethod
    def getDayDelta(endDate, startYear):
        """
        Take a day
        """
        startYear = date(startYear, 1, 1)
        delta = endDate - startYear
        return delta.days 