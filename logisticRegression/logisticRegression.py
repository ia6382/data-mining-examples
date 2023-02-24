import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    #g(theta'x)
    #X je lahko tudi matrika,le prej mora biti transponirana
    p = 1 / (1 + np.exp(-(np.dot(theta.T, x))))

    return p


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije (verjetje l z regularizacijo).
    """
    (m,n) = np.shape(X)

    #izogni se log(0) -> zamenjaj nule z majhnim stevilom
    htmp = h(X.T, theta)
    htmp1 = 1 - h(X.T, theta)
    htmp[htmp == 0] = 1e-10
    htmp1[htmp1 == 0] = 1e-10

    l = np.dot(y, np.log(htmp).T) + np.dot((1 - y), np.log(htmp1).T)
    regularizacija = lambda_ * 0.5 * np.dot(theta[1:], theta[1:]) #theta[0] ne smemo regularizirati

    lr = (-l + regularizacija)/m #mogoce pravilno -(l+regularizacija)/m
    return lr;


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    (m,n) = np.shape(X)

    dl = np.dot((y - h(X.T, theta)), X)
    dr = lambda_*theta
    dr[0] = 0 #theta[0] ne smemo regularizirati

    grad = (-dl + dr)/m #mogoce pravilno -(l+regularizacija)/m
    return grad

def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    eps = 1e-5
    (m,n) = np.shape(X)

    # sredinska formula, eps mora biti dobro določen
    grad = np.zeros(n)
    for j in range(n):
        epsVector = np.zeros(n)
        epsVector[j] = eps
        grad[j] = (cost(theta + epsVector, X, y, lambda_) - cost(theta - epsVector, X, y, lambda_)) / (2*eps)

    return grad


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetnost razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    (m,n) = np.shape(X)
    predikcijaY = np.zeros((m, 2))
    urejenaPredikcijaY = np.zeros((m, 2))

    #permutiraj vrstice in shrani indekse
    np.random.seed(0) #fiksiraj seme, da lahko primerjamo rezultate za testiranje regularizacije
    indeksi = np.random.permutation(m) # DOC: If x is an integer, randomly permute np.arange(x)

    #razdeli podatke na k delov
    while (m % k) != 0: #ce delitev ni mozna 
        k = k-1;
    deli = np.split(indeksi, k)

    #vsak del podatkov bo bil enkrat testna mnozica, ostali so skupaj ucna mnozica
    for i in range(k):
        d = deli[i]
        ostali = np.setdiff1d(indeksi, d, assume_unique=True)

        #doloci ucno in testno mnozico    
        ucnaX = X[ostali,:]
        testnaX = X[d,:]
        ucnaY = y[ostali]
        testnaY = y[d]

        #nauci model na ucni mnozici
        model = learner(ucnaX, ucnaY)
        #testiraj model na testni mnozici
        predikcija = [model(x) for x in testnaX]

        #dodaj predikcijo v skupni vektor predikcij
        begin = int(i*(m/k))
        end = int((i+1)*(m/k))
        predikcijaY[begin:end] = predikcija

    #uredi predikcije po začetnem vrstnem redu
    for i, orig in enumerate(indeksi):
        urejenaPredikcijaY[orig] = predikcijaY[i]

    return urejenaPredikcijaY

def R2(real, predictions):
    #upostevaj vecjo verjetnost
    p = np.zeros(len(real))
    for i in range(len(real)):
        if predictions[i][0] >= predictions[i][1]:
            p[i] = 0
        else:
            p[i] = 1

    meanReal = sum(real)/len(real)

    err = sum((real - p)**2)
    sse = sum((real - meanReal)**2)
    R2 = 1 - (err/sse)
    return R2


def CA(real, predictions):
    #delež koliko primerov smo pravilno uvrstili
    pravilni = 0
    for (p, r) in zip(predictions, real):
        razred = 0
        if p[0] <= p[1]:
            razred = 1

        if razred == r:
            pravilni += 1

    ca = pravilni/len(real)
    return ca

def optimalLambda(testFun, measureFun, X, y, lambdas):
    #najdi optimalno lambdo z križnim primerjanjem

    cMax = float('-inf')
    optimalLambda = 0

    for l in lambdas:
        learner = LogRegLearner(lambda_=l)
        predictions = testFun(learner, X, y)
        c = measureFun(y, predictions)
        if c > cMax:
            cMax = c
            optimalLambda = l

    return optimalLambda


if __name__ == "__main__":
    pass

