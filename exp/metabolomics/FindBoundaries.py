import numpy
from exp.metabolomics.MetabolomicsUtils import MetabolomicsUtils

X, X2, df = MetabolomicsUtils.loadData()

#Just figure out the boundaries of the levels 
numpy.set_printoptions(threshold=3000)
labelNames = ["IGF1.val", "Cortisol.val", "Testosterone.val"]
labelNames2 = ["Ind.IGF1.1", "Ind.IGF1.2", "Ind.IGF1.3"]
YList = MetabolomicsUtils.createLabelList(df, labelNames)
YList2 = MetabolomicsUtils.createLabelList(df, labelNames2)

Y, inds = YList[0]
Y1 = numpy.array(df.rx(labelNames2[0])).ravel()[inds]
Y2 = numpy.array(df.rx(labelNames2[1])).ravel()[inds]
Y3 = numpy.array(df.rx(labelNames2[2])).ravel()[inds]

inds = numpy.argsort(Y)
YY = numpy.c_[Y[inds], Y1[inds]]
YY = numpy.c_[YY, Y2[inds]]
YY = numpy.c_[YY, Y3[inds]]
print(YY)

labelNames2 = ["Ind.Cortisol.1", "Ind.Cortisol.2", "Ind.Cortisol.3"]
YList2 = MetabolomicsUtils.createLabelList(df, labelNames2)

Y, inds = YList[1]
Y1 = numpy.array(df.rx(labelNames2[0])).ravel()[inds]
Y2 = numpy.array(df.rx(labelNames2[1])).ravel()[inds]
Y3 = numpy.array(df.rx(labelNames2[2])).ravel()[inds]

inds = numpy.argsort(Y)
YY = numpy.c_[Y[inds], Y1[inds]]
YY = numpy.c_[YY, Y2[inds]]
YY = numpy.c_[YY, Y3[inds]]
print(YY)

#Only testorone categorises the levels correctly
labelNames2 = ["Ind.Testo.1", "Ind.Testo.2", "Ind.Testo.3"]
YList2 = MetabolomicsUtils.createLabelList(df, labelNames2)

Y, inds = YList[2]
Y1, inds = YList2[0]
Y2, inds = YList2[1]
Y3, inds = YList2[2]

print((numpy.min(Y[Y1==1]), numpy.max(Y[Y1==1])))
print((numpy.min(Y[Y2==1]), numpy.max(Y[Y2==1])))
print((numpy.min(Y[Y3==1]), numpy.max(Y[Y3==1])))

#Results (figures are exclusive)
"""
Cortisol: 0 - 88 (-1), 89 - 224 (0), 225 - 572 (1) 
IGF1: 0 - 199 (-1), 200-440 (0), 441 - 781 (1)
Testosterone: 0-3 (-1), 3-9 (0), 9 - 12.91 (1)
"""