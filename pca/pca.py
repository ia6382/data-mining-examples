from unidecode import unidecode
import numpy as np
from matplotlib import pyplot as plt 
import math
import os
import re
import random
import unittest

def prepare_data_matrix():
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    numOfTerms = 100
    languages = []
    trojice = {} #vse mozne trojice

    #naredi strukturo DATA - slovar jezikov(datotek), ki vsebujejo slovar stevila pojavitve trojic
    #socasno se stej v kolikih datotekah se pojavi posamezna trojica
    DATA = {}
    direktorij = os.fsencode("./HRlanguages/")
    numOfFiles = len(os.listdir(direktorij))
    for datoteka in os.listdir(direktorij):
        #odpri in obdelaj datoteko - prevod
        kmersFreq = obdeljaDatoteko(os.path.join(direktorij, datoteka), trojice)

        #dodaj v DATA
        name = os.fsdecode(datoteka)[:-4]
        DATA[name] = kmersFreq

        languages.append(name)

    #uredi trojice po idf meri
    trojiceList = []
    for (k, v) in trojice.items():
        idf = math.log(numOfFiles/v)
        trojiceList.append((k, idf))

    trojiceList = sorted(trojiceList, key=lambda x: x[1])[:numOfTerms]

    #zgradi matriko X
    X = np.zeros((numOfFiles, numOfTerms))
    for i in range(numOfFiles):
        for j in range(numOfTerms):
            jezik = languages[i]
            trojica = trojiceList[j][0]
            if trojica in DATA[jezik]:
                X[i][j] = DATA[jezik][trojica] #frekvenca trojice v dokumentu jezika

    #centriraj podatke
    #mean = np.mean(X, axis=0) #povprecje po stolpcih
    #X = X - mean

    return X, languages

def obdeljaDatoteko(file, trojice):
    """vrni slovar pojavitva trojic"""

    f = open(file, "rt", encoding="utf8")
    s = f.read()
    f.close()
    s = unidecode(s)
    s = s.lower()
    s = re.sub(r"[^A-Z a-z]+", '', s) #regularni izraz, vse kar ni crka in presledek zamenja z praznim stringom
    s = re.sub(r"[ ]{2,}", ' ', s) #mogoce potrebujemo informacijo o besedah ne samo črkah brez presledki
        
    #prestej vse trojice v prevodu in stej v kolikoh datotekah se pojavi posamezna trojica
    k = 3
    kmersFreq = {}
    for i in range(len(s[:-(k-1)])):
        kmer = s[i:i+k]
        if kmer in kmersFreq:
            kmersFreq[kmer] += 1
        else:
            kmersFreq[kmer] = 1
            #trojica obstaja v tem dokumentu
            if kmer in trojice: 
                trojice[kmer] += 1 
            else:
                trojice[kmer] = 1
    
    return kmersFreq

def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    (m,n) = np.shape(X)
    if m != n: #hack za avtomatski test_power_iteration_single, ki pracakuje da cov(X) izracunamo zntraj te funkcije
        covX = np.cov(X.T)
    else:
        covX = X #kasneje za Hotelling deflation potrebujemo cov(X) zunaj funkcije

    precision = 1e-10
    maxIter = 50
    x = np.random.rand(covX.shape[1]) #,1 da je stolpec
    
    #uporabimo neskoncno normo (max)
    ls = np.max(x)
    for i in range(maxIter):
        x = np.dot(covX, x)
        ln = np.max(x)
        x = np.divide(x,ln)
        if (abs(ln-ls) < precision):
           break 
        ls = ln

    eigenvalue = np.max(np.dot(covX,x))
    x = np.divide(x,np.linalg.norm(x)) #ogrodje zahteva 2.normo
    eigenvector = x

    return (eigenvector, eigenvalue)


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    (m, n) = np.shape(X)
    eigenvectors = np.zeros((2, n))
    eigenvalues = np.zeros((2))

    #najdi prvi lastni vektor
    covX = np.cov(X.T)
    (eVec1, eVal1) = power_iteration(covX)
    """
    #projeciramo vse vrstice na lastni vektor
    a = np.vstack(eVec1)
    projX = np.zeros((m,n))
    for i in range(m):
        b = np.vstack(X[i,:])
        projX[i,:] = project(a, b)
    #odstejemo projekcijo od podatkov
    orthX = X - projX
    #rezultat je projekcija na hiperravnino, ki je pravokotna lastnemu vektorju
    """
    #Hotelling deflation - potrebuje simetricno matriko (cov)
    a = np.vstack(eVec1)
    #projX = eVal1*((np.dot(a, a.T))/(np.dot(a.T, a)))
    projX = eVal1*(np.dot(a, a.T)) #ni dejanska projekcija, le imena spremenljivk so ista
    orthX = covX - projX #defaltali (dali na 0) smo največjo (prvo najdeno) lastno vrednost
    
    #izracunaj 2. lasnti vektor
    (eVec2, eVal2) = power_iteration(orthX)

    #vrni rezultat
    eigenvectors[0, :] = eVec1
    eigenvectors[1, :] = eVec2
    eigenvalues[0] = eVal1
    eigenvalues[1] = eVal2
    return (eigenvectors, eigenvalues)

"""def project(a, b):
    pa = np.dot((np.dot(a.T, b)/np.dot(b.T, b)).item(), b)
    pa = np.array(pa.T)[0]
    return pa"""

def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    #centriraj podatke
    mean = np.mean(X, axis=0) #povprecje po stolpcih
    X = X - mean

    #projekcija na vektorja
    P = np.dot(X,vecs.T)

    return P
    

def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return eigenvalues.sum()/total_variance(X)

if __name__ == "__main__":
    # prepare the data matrix
    X, languages = prepare_data_matrix()

    # PCA
    (eigenvectors, eigenvalues) = power_iteration_two_components(X)
    P = project_to_eigenvectors(X, eigenvectors)
    r = explained_variance_ratio(X, eigenvectors, eigenvalues)

    # plotting
    plt.scatter(P[:,0], P[:,1], c='orange')
    for i in range(X.shape[0]):
        language = languages[i]
        plt.text(P[i,0], P[i,1], language, fontsize=10)
    plt.xlabel("1. glavna komponenta")
    plt.ylabel("2. glavna komponenta")
    plt.title("Razložena varianca: "+str(round(r, 3)))
    plt.show()
