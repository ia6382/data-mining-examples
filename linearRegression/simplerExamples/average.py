"""
Control classifier that predicts bus arrival times just based on a simple average.
"""

import gzip
import numpy
import csv
import lpputils

from collections import defaultdict

def linekey(d):
    return tuple(d[2:5])

class SeparateBySetLearner(object):

    def __init__(self, base):
        self.base = base

    def __call__(self, data):
        rsd = defaultdict(list)
        rsc = {}
        #separate different bus lines
        for d in data:
            rsd[linekey(d)].append(d)

        #build a prediction model for each line
        #with self.base
        for k in rsd:
            cl = self.base(rsd[k])
            rsc[k] = cl
        return SeparateBySetClassifier(rsc)

class SeparateBySetClassifier(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def __call__(self, x):
        #pass to input to the correct classifier for that line
        try:
            return self.classifiers[linekey(x)](x)
        except:
            # a new line: make an average
            return numpy.mean([c(x) for c in self.classifiers.values()])

class AverageTripLearner(object):

    def __call__(self, data):
        delays = [ lpputils.tsdiff(d[-1], d[-3]) for d in data ]
        mean = sum(delays)/len(delays)

        return AverageTripClassifier(mean)

class AverageTripClassifier(object):
    
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, x):
        #does not use the input example at all, because 
        #in this case the prediction is always the same
        return self.mean

if __name__ == "__main__":

    f = gzip.open("train.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [ d for d in reader ]

    #model each line separately with AverageTripLearner
    l = SeparateBySetLearner(AverageTripLearner())
    c = l(data)

    for d in data:
        pass
        #print linekey(d), c(d)

    f = gzip.open("test.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader) #skip legend

    fo = open("average2.txt", "wt")
    for l in reader:
        fo.write(lpputils.tsadd(l[-3], c(l)) + "\n")
    fo.close()
