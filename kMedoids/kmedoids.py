from unidecode import unidecode
from matplotlib import pyplot as plt 
import math
import os
import re
import random

def obdeljaDatoteko(file):
    #vrni slovar pojavitva trojic

    f = open(file, "rt", encoding="utf8")
    s = f.read()
    f.close()
    s = unidecode(s)
    s = s.lower()
    s = re.sub(r"[^A-Z a-z]+", '', s) #regularni izraz, vse kar ni crka in presledek zamenja z praznim stringom
    s = re.sub(r"[ ]{2,}", ' ', s) #mogoce potrebujemo informacijo o besedah ne samo črkah brez presledki
    #s = re.sub(r"[^A-Za-z]+", '', s) #brez presledkov
        
    #prestej trojice za vsak prevod
    k = 3
    kmersFreq = {}
    for i in range(len(s[:-(k-1)])):
        kmer = s[i:i+k]
        if kmer in kmersFreq:
            kmersFreq[kmer] += 1
        else:
            kmersFreq[kmer] = 1
    
    return kmersFreq

def obdelajPrevode(pot):
    #naredi strukturo DATA za grucenje - slovar parov slovarja in int {(jezik1:{trojica1: pojavitve, ...}, indeks), ...}

    DATA = {}
    direktorij = os.fsencode(pot)
    for indeks, datoteka in enumerate(os.listdir(direktorij)):
        #odpri in obdelaj datoteko - prevod
        kmersFreq = obdeljaDatoteko(os.path.join(direktorij, datoteka))

        #dodaj v DATA
        DATA[os.fsdecode(datoteka)[:-4]] = (kmersFreq, indeks) #(slovar trojic, indeks v matriki razdalj)
        
    return DATA

def kosRazdalja(dict1, dict2):
    #izracunaj kosinusno razdaljo med dvema prevodoma (slovarjema trojk in njihovih pojavitev)

    normDict1, normDict2 = 0, 0
    for i in dict1.values():
        normDict1 += i**2
    normDict1 = math.sqrt(normDict1)
    for i in dict2.values():
        normDict2 += i**2
    normDict2 = math.sqrt(normDict2)

    skalarniProdukt = 0
    if len(dict1) > len(dict2): #ce ima dict1 vec trojic gremo tako cez vse skupne z dict2
        for (k1, v1) in dict1.items():
            if k1 in dict2:
                #skalarniProdukt += (v1/normDict1)*(dict2[k1]/normDict2)
                skalarniProdukt += v1*dict2[k1]
    else: #ce ima dict2 vec trojic gremo tako cez vse skupne z dict1
        for (k2, v2) in dict2.items():
            if k2 in dict1:
                #skalarniProdukt += (v2/normDict2)*(dict1[k2]/normDict1)
                skalarniProdukt += v2*dict1[k2]

    kosPodobnost = skalarniProdukt/(normDict1*normDict2)
    return (1 - kosPodobnost) #kosinusna razdalja

def narediMatrikoRazdalj(DATA):
    #definiraj len*len matriko
    l = len(DATA)
    matrika = [[0] * l for i in range(l)]

    #izracunaj razdalja med vsemi pari tock
    for (vrstica, i) in enumerate(DATA.values()):
        for (stolpec, j) in enumerate(DATA.values()):
            #matrika razdalj je simetricna - izracunaj le polovico
            if stolpec > vrstica:
                razdalja = kosRazdalja(i[0], j[0])
                matrika[i[1]][j[1]] = razdalja
            elif stolpec == vrstica:
                matrika[i[1]][j[1]] = 0.0
            else:
                matrika[i[1]][j[1]] = matrika[j[1]][i[1]]

    return matrika

def konfiguracija(DATA, matrikaRazdalj, medoids, k):
    #povezi tocke v skupine z najblizjimi medoidi
    skupine = [[] for i in range(k)]

    for (dk, dv) in DATA.items():
        minRazdalja = float('inf')
        idSkupine = None
        
        for (mk, mv) in medoids.items():
            indTocke = dv[1]
            indMedoida = DATA[mk][1]
            razdalja = matrikaRazdalj[indTocke][indMedoida]

            if(razdalja < minRazdalja):
                minRazdalja = razdalja
                idSkupine = mv
        
        skupine[idSkupine].append([dk, minRazdalja])
    
    #izracunaj ceno konfiguracije
    cena = 0
    for i in skupine:
        for j in i:
            cena += j[1]
    
    return (skupine, cena)


def kMedoids(DATA, k, matrikaRazdalj):
    #iz DATA izberi k RAZLICNIH nakjlucnih medoidov
    medoids = {}
    i = 0
    while i < k:
        medoid = random.choice(list(DATA.keys()))
        if not(medoid in medoids):
            medoids[medoid] = i
            i += 1
        
    #razvrsti tocke v skupine in izracunaj ceno konfiguracije
    (skupine, cena) = konfiguracija(DATA, matrikaRazdalj, medoids, k)

    #iterativno izboljsuj konfiguracijo
    while True:
        #print("****")
        najboljsaCena = float('inf')
        najboljsiMedoidi = None
        najboljseSkupine = None

        #za vse medoide in tocke, ki niso medoidi
        for (mk, mv) in medoids.items():
            for (dk, dv) in DATA.items():
                if not(dk in medoids):

                    #oceni zamenjavo tocke dk z medoidom mk
                    potencialniMedoidi = medoids.copy()
                    del potencialniMedoidi[mk]
                    potencialniMedoidi[dk] = mv

                    (potencialneSkupine, potencialnaCena) = konfiguracija(DATA, matrikaRazdalj, potencialniMedoidi, k)

                    #zapomni si najboljso zamenjavo
                    if potencialnaCena < najboljsaCena:
                        najboljsaCena = potencialnaCena
                        najboljsiMedoidi = potencialniMedoidi.copy()
                        najboljseSkupine = potencialneSkupine
            
        #zares naredi najboljso zamenjavo
        #ampak le ce je cena boljsa kot v prejsni iteraciji
        if najboljsaCena < cena:
            cena = najboljsaCena
            medoids = najboljsiMedoidi.copy()
            skupine = najboljseSkupine
        else: #ce ni prekini izvajanje
            break
        
    return (skupine, medoids)

def izpisiSkupine(skupine):
    for (st, i) in enumerate(skupine):
        print("skupina "+str(st))
        for j in i:
            if(j[1] <= 0.001 and j[1] >= -0.001):
                print("  *"+j[0]+" "+str(round(j[1], 2))+"* ")
            else:
                print("  "+j[0]+" "+str(round(j[1], 2)))

def oceniSilhuetoRazbitja(DATA, skupine, matrikaRazdalj):
    s = 0 #silhueta celotnega razbitja

    for (ind, c) in enumerate(skupine):
        for i in c:
            if len(c) == 1:
                continue

            #a clen
            a_i = 0
            #razdalja do vseh ostalih primerov v skupini
            for j in c:
                jIme = j[0]
                iIme = i[0]
                if jIme != iIme:
                    indMatrikej = DATA[jIme][1]
                    indMatrikei = DATA[iIme][1]
                    a_i += matrikaRazdalj[indMatrikei][indMatrikej]
            a_i = a_i/(len(c)-1)

            #b clen
            #minimalna razdalja do vseh ostalih primerov izven skupine
            bMin = float('inf')
            for (indk, ck) in enumerate(skupine):
                if indk != ind:
                    b_i = 0
                    for j in ck:
                        indMatrikej = DATA[j[0]][1]
                        indMatrikei = DATA[i[0]][1]
                        b_i += matrikaRazdalj[indMatrikei][indMatrikej]
                    b_i = b_i/len(ck) 

                    if(b_i < bMin):
                        bMin = b_i

            #s_i = silhueta za primer i
            s_i = (bMin - a_i)/max(a_i, bMin)
            s += s_i
    
    s = s/len(DATA)
    return s

def histogramSilhuet(DATA, matrikaRazdalj, k, n):
    #ponovi grucenje s k medoidi nkrat in izrisi histogram silhiet razbitja
    silhuete = [0]*n
    minS = float('inf')
    minSkupine = None
    maxS = float('-inf')
    maxSkupine = None

    for i in range(n):
        (skupine, _) = kMedoids(DATA, k, matrikaRazdalj)
        s = oceniSilhuetoRazbitja(DATA, skupine, matrikaRazdalj)
        silhuete[i] = s
        
        #shrani najvecjo in najmanso silhueto in skupine
        if s < minS:
            minS = s
            minSkupine = skupine
        if s > maxS:
            maxS = s
            maxSkupine = skupine

    #izpisi skupine
    print("--- "+str(minS)+" ---")
    izpisiSkupine(minSkupine)
    print("\n\n--- "+str(maxS)+" ---")
    izpisiSkupine(maxSkupine)

    #narisi histogram
    plt.hist(silhuete, bins=5) #density="norm"
    plt.xlabel("silhueta razbitja")
    plt.ylabel("frekvenca")
    plt.title("Porazdelitev (histogram) vrednosti za "+str(n)+" silhuet")
    plt.show()

def prepoznajJezik(file, DATA):
    neznaniJezik = obdeljaDatoteko(file)

    #izracunaj razdalje do vseh jezikov v DATA  
    razdalje = []
    for k, v in DATA.items():
        d = kosRazdalja(neznaniJezik, v[0])
        razdalje.append((k, d))
    
    #uredi razdalje
    razdalje = sorted(razdalje, key=lambda x: x[1])

    #izpisi top 3 veretnosti
    #verjentost = podobnost*100 = (1-razdalja)*100
    [print("    "+jezik+" "+str(round((1-d)*100, 3))) for (jezik, d) in razdalje[:3]]

def prepoznajJezikOpt(file, DATA, matrikaRazdalj, skupine, medoids):
    #optimizacija - isci le v skupini, ki ima najbolj podoben medoid
    #optimizacija je le ce smo prej ze naredili razvrscanje v skupine
    neznaniJezik = obdeljaDatoteko(file)

    #poisci najblizji medoid in pripadajoco skupino
    minD = float('inf')
    indS = None
    for k, v in medoids.items():
        d = kosRazdalja(neznaniJezik, DATA[k][0])
        if d < minD:
            minD = d
            indS = v

    #posici razdalje v skupini medoida
    razdalje = []
    for (jezik, _) in skupine[indS]:
        d = kosRazdalja(DATA[jezik][0], neznaniJezik)
        razdalje.append((jezik, d))
    
    #uredi razdalje
    razdalje = sorted(razdalje, key=lambda x: x[1])

    #izpisi top 3 veretnosti
    #verjentost = podobnost*100 = 1-razdalja*100
    [print(jezik+" "+str(round((1-d)*100, 3))) for (jezik, d) in razdalje[:3]]

def avtomatizirajPrepoznavo(pot, DATA):
    #prepoznaj jezik vseh datotek v pot
    direktorij = os.fsencode(pot)
    for datoteka in os.listdir(direktorij):
        print("\n"+os.fsdecode(datoteka))
        prepoznajJezik(os.path.join(direktorij, datoteka), DATA)

if __name__ == "__main__":
    DATA = obdelajPrevode("data/HRtranslations/")

    matrikaRazdalj = narediMatrikoRazdalj(DATA)
    histogramSilhuet(DATA, matrikaRazdalj, 5, 100)
    
    #avtomatizirajPrepoznavo("data/excerpts/", DATA)

    #neuporabljena prepoznava
    #(skupine, medoids) = kMedoids(DATA, 5, matrikaRazdalj)
    #prepoznajJezikOpt("italtest.txt", DATA, matrikaRazdalj, skupine, medoids)
    