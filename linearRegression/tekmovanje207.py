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
        Napovej razred za vektor vrednosti znacilk.
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
    znacilke = definirajZnacilke()

    #pridobi primere  
    (linije, _, _, _) = parseData(znacilke, ftrain, -1) #ne testiraj na nobenem mesecu v ucnih podatkih
    print("\t1")

    #nauci modele
    modeli = {}
    for (k, v) in linije.items():
        Xucna = v[0]
        yucna = v[1]
        Xucna = sp.csr_matrix(Xucna)
        yucna = np.array(yucna)
    
        lr = LinearLearner(lambda_=l) #init model (samo nastavi lambdo)
        linear = lr(Xucna, yucna) #nauci model
        modeli[k] = linear
    print("\t2")

    #testiraj modele
    (_, Xtest, _, odhodi) = parseData(znacilke, ftest, 12)
    ynapoved = []
    for (x, o) in zip(Xtest, odhodi):
        if x[0] == '1MESTNI LOG - VIŽMARJE  VIŽMARJE; sejemTbilisijskaVIŽMARJE':
            linear = modeli['1MESTNI LOG - GARAŽA  GARAŽAKoprskaRemiza']
        elif x[0] == '1VIŽMARJE - MESTNI LOG  MESTNI LOG; sejemŠentvidMESTNI LOG':
            linear = modeli['1VIŽMARJE - MESTNI LOG  MESTNI LOGŠentvidMESTNI LOG']
        else:
            linear = modeli[x[0]]

        x = np.array(x[1])
        ynapoved.append(linear(x))

    ynapoved = np.array(ynapoved)
    print("\t3")

    zapisiNapovedi(odhodi, ynapoved)

    
def testirajVseMeseceLokalno(f, l):
    #definiraj znacilke
    znacilke = definirajZnacilke()
    print("znacilke definirane")
    mae = 0
    for i in range(1,12):
        print("testiranje: "+str(i))

        #pridobi podatke
        (linije, Xtest, ytest, odhodi) = parseData(znacilke, f, i)
        print("\t1")

        #nauci modele
        modeli = {}
        for (k, v) in linije.items():
            Xucna = v[0]
            yucna = v[1]
            Xucna = sp.csr_matrix(Xucna)
            yucna = np.array(yucna)
        
            lr = LinearLearner(lambda_=l) #init model (samo nastavi lambdo)
            linear = lr(Xucna, yucna) #nauci model
            modeli[k] = linear
        print("\t2")

        #testiraj modele
        ynapoved = []
        for (x, o) in zip(Xtest, odhodi):
            linear = modeli[x[0]]
            x = np.array(x[1])
            ynapoved.append(linear(x))

        ynapoved = np.array(ynapoved)
        print("\t3")

        e = meanAbsoluteError(ytest, ynapoved)
        mae += e
        print("\tmae: "+str(e))

    print("MAE: "+str(mae/11))

def definirajZnacilke():
    znacilke = {'pocitnice': 0, "delovnik": 1, "sob":2, "nedAliPraznik": 3}
    n = len(znacilke)

    for i in range (24):
        znacilke["ura"+str(i)] = n
        n += 1
    
    print(znacilke)
    return znacilke

def parseData(znacilke, file_name, testMonth):
    linije = {} #slovar ki ima za vsako linijo ucno X in ucno y
    
    #seznami za testiranje
    Xtest = []
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
            vec[znacilke["pocitnice"]] = 1
        if (str(date.day)+"."+str(mesec)+".") in ["1.1.", "2.1.", "8.2.", "27.4.", "1.5.", "2.5.", "8.6.", "25.6.", 
            "15.8.", "17.8.", "15.9.", "25.10.", "31.10.", "1.11.", "23.11.", "25.12.", "26.12."] or dan == 6:
            vec[znacilke["nedAliPraznik"]] = 1
        elif dan == 5:
            vec[znacilke["sob"]] = 1
        else:
            vec[znacilke["delovnik"]] = 1
        
        vec[znacilke["ura"+str(ura)]] = 1

        #dadoj vrednosti v primerne sezname
        if mesec == testMonth:
            Xtest.append((idLinije, vec))
            if mesec != 12: # ce je 12 nimamo znanih rezultatov
                ytest.append(tsdiff(casPrihoda, casOdhoda))
            odhodi.append(casOdhoda);
        else:
            if idLinije in linije:
                linije[idLinije][0].append(vec) 
                linije[idLinije][1].append(tsdiff(casPrihoda, casOdhoda))
            else:
                linije[idLinije] = [[vec], [tsdiff(casPrihoda, casOdhoda)]]
            
    f.close()
    return (linije, Xtest, ytest, odhodi)


if __name__ == "__main__":
    #napovejZaTekmovanje("./train.csv", "./test.csv", 0.5)
    testirajVseMeseceLokalno("./train.csv", 0.5)
    