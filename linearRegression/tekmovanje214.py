import datetime
import csv
from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse
import time

import scipy.sparse as sp
import numpy as np

"""
LPPUTILS - branje datuma/casa
"""
FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def parsedate(x):
    if not isinstance(x, datetime.datetime):
        x = datetime.datetime.strptime(x, FORMAT)
    return x

def tsdiff(x, y):
    return (parsedate(x) - parsedate(y)).total_seconds()

def tsadd(x, seconds):
    d = datetime.timedelta(seconds=seconds)
    nd = parsedate(x) + d
    return nd.strftime(FORMAT)

"""
LINEAR - linearna regresija
"""
def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


def hl(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)

def cost_grad_linear(theta, X, y, lambda_):
    # do not regularize the first element
    sx = hl(X, theta)
    j = 0.5 * numpy.mean((sx - y) * (sx - y)) + 1 / 2. * lambda_ * theta[1:].dot(theta[1:]) / y.shape[0]
    grad = X.T.dot(sx - y) / y.shape[0] + numpy.hstack([[0.], lambda_ * theta[1:]]) / y.shape[0]
    return j, grad


class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = append_ones(X)

        th = fmin_l_bfgs_b(cost_grad_linear,
                           x0=numpy.zeros(X.shape[1]),
                           args=(X, y, self.lambda_))[0]

        return LinearRegClassifier(th)


class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        return hl(x, self.th)

"""
OBDELOVANJE PODATKOV
"""
def meanAbsoluteError(y1, y2):
    a = abs(y2 - y1)
    mae = np.mean(a)
    return mae

def zapisiNapovedi(odhodi, trajanjeNapovedi):
    f = open("napoved.txt", "wt")
    for i, cas in enumerate(odhodi):
        f.write(tsadd(cas, trajanjeNapovedi[i])+"\n")
    f.close()

def napovejZaTekmovanje(ftrain, ftest, l):
    #definiraj znacilke
    znacilke = definirajZnacilke([ftrain, ftest])

    #pridobi primere  
    (Xucna, yucna, _, _, _) = parseData(znacilke, ftrain, -1) #ne testiraj na nobenem mesecu v ucnih podatkih
    Xucna = sp.csr_matrix(Xucna)
    yucna = np.array(yucna)
    print(1)

    #nauci model
    lr = LinearLearner(lambda_=l) #init model (samo nastavi lambdo)
    linear = lr(Xucna, yucna) #nauci model
    print(2)

    #testiraj model
    (_, _, Xtest, _, odhodi) = parseData(znacilke, ftest, 12)
    Xtest = np.array(Xtest)
    ynapoved = [linear(x) for x in Xtest]
    ynapoved = np.array(ynapoved)
    print(3)

    zapisiNapovedi(odhodi, ynapoved)

def testirajVseMeseceLokalno(f, l):
    #definiraj znacilke
    znacilke = definirajZnacilke([f])
    print("znacilke definirane")
    mae = 0
    for i in range(1,12):
        print("testiranje: "+str(i))
        #pridobi podatke
        (Xucna, yucna, Xtest, ytest, odhodi) = parseData(znacilke, f, i)
        Xucna = sp.csr_matrix(Xucna)
        yucna = np.array(yucna)
        Xtest = np.array(Xtest)
        ytest = np.array(ytest)
        print("\t1")
        #nauci model
        lr = LinearLearner(lambda_=l) #init model (samo nastavi lambdo)
        linear = lr(Xucna, yucna) #nauci model
        print("\t2")
        #testiraj model
        ynapoved = [linear(x) for x in Xtest]
        ynapoved = np.array(ynapoved)
        print("\t3")
        e = meanAbsoluteError(ytest, ynapoved)
        mae += e
        print("\tmae: "+str(e))

    print("MAE: "+str(mae/11))

def definirajZnacilke(files):
    obdelani = set()
    osnovneZnacilke = ["pocitnice", "delovnik", "sob", "nedAliPraznik", "ura", "ura2", "ura3"]
    znacilke = {}
    n = len(znacilke)
    
    for file_name in files:
        f = open(file_name, "rt", newline='', encoding="utf8")
        next(f) #preskoci glavo
        for line in csv.reader(f, delimiter='\t'):
            #registracija = line[0].split("-")[1]
            #voznik = line[1]
            #opis = line[5]
            #zadnjaPostaja = line[7]

            #pridobi celotni id linije
            idLinije = line[2]+line[3]+line[4]+line[5]+line[7]

            if not(idLinije in obdelani):
                obdelani.add(idLinije)
                for i in osnovneZnacilke:
                    znacilke[idLinije+i] = n
                    n += 1

    print(znacilke)
    return znacilke

def parseData(znacilke, file_name, testMonth):
    #naredi matriko primerov s pridobljenimi znacilkami
    Xucna = []
    Xtest = []
    yucna = []
    ytest = []
    odhodi = []
    n = len(znacilke)
    
    f = open(file_name, "rt", newline='', encoding="utf8")
    next(f) #preskoci glavo
    i = 0
    for line in csv.reader(f, delimiter='\t'):
        vec = [0]*n

        #pridobi celotni id linije
        idLinije = line[2]+line[3]+line[4]+line[5]+line[7]
            
        #pridobi znacilke iz datuma in casa
        casOdhoda = line[6]
        casPrihoda = line[8]
        date = parsedate(casOdhoda)
        dan = date.weekday()
        mesec = date.month
        ura = date.hour

        #zapisi vrednosti znacilk
        if mesec in [6,7,8]: 
            vec[znacilke[idLinije+"pocitnice"]] = 1
        if (str(date.day)+"."+str(mesec)+".") in ["1.1.", "2.1.", "8.2.", "27.4.", "1.5.", "2.5.", "8.6.", "25.6.", 
            "15.8.", "17.8.", "15.9.", "25.10.", "31.10.", "1.11.", "23.11.", "25.12.", "26.12."] or dan == 6:
            vec[znacilke[idLinije+"nedAliPraznik"]] = 1
        elif dan == 5:
            vec[znacilke[idLinije+"sob"]] = 1
        else:
            vec[znacilke[idLinije+"delovnik"]] = 1
        vec[znacilke[idLinije+"ura"]] = (ura/24)
        vec[znacilke[idLinije+"ura2"]] = (ura/24)**2
        vec[znacilke[idLinije+"ura3"]] = (ura/24)**3

        #dodaj vrednosti v primerne sezname
        if mesec == testMonth:
            Xtest.append(vec)
            if mesec != 12: # ce je 12 nimamo znanih rezultatov
                ytest.append(tsdiff(casPrihoda, casOdhoda))
            odhodi.append(casOdhoda);
        else:
            Xucna.append(vec) 
            yucna.append(tsdiff(casPrihoda, casOdhoda))
            
    f.close()
    return (Xucna, yucna, Xtest, ytest, odhodi)


if __name__ == "__main__":
    #napovejZaTekmovanje("./tekmovanje/train.csv", "./tekmovanje/test.csv", 0.5)
    testirajVseMeseceLokalno("./tekmovanje/train.csv", 0.5)
    