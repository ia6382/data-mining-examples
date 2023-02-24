from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse

import scipy.sparse as sp
import numpy as np


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


if __name__ == "__main__":
    
    X = numpy.array([[6.3200e-03, 1.8000e+01, 2.3100e+00, 0.0000e+00, 5.3800e-01,
                      6.5750e+00, 6.5200e+01, 4.0900e+00, 1.0000e+00, 2.9600e+02,
                      1.5300e+01, 3.9690e+02, 4.9800e+00],
                     [2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,
                      6.4210e+00, 7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,
                      1.7800e+01, 3.9690e+02, 9.1400e+00],
                     [2.7290e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,
                      7.1850e+00, 6.1100e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,
                      1.7800e+01, 3.9283e+02, 4.0300e+00],
                     [3.2370e-02, 0.0000e+00, 2.1800e+00, 0.0000e+00, 4.5800e-01,
                      6.9980e+00, 4.5800e+01, 6.0622e+00, 3.0000e+00, 2.2200e+02,
                      1.8700e+01, 3.9463e+02, 2.9400e+00],
                     [6.9050e-02, 0.0000e+00, 2.1800e+00, 0.0000e+00, 4.5800e-01,
                      7.1470e+00, 5.4200e+01, 6.0622e+00, 3.0000e+00, 2.2200e+02,
                      1.8700e+01, 3.9690e+02, 5.3300e+00],
                     [2.9850e-02, 0.0000e+00, 2.1800e+00, 0.0000e+00, 4.5800e-01,
                      6.4300e+00, 5.8700e+01, 6.0622e+00, 3.0000e+00, 2.2200e+02,
                      1.8700e+01, 3.9412e+02, 5.2100e+00],
                     [8.8290e-02, 1.2500e+01, 7.8700e+00, 0.0000e+00, 5.2400e-01,
                      6.0120e+00, 6.6600e+01, 5.5605e+00, 5.0000e+00, 3.1100e+02,
                      1.5200e+01, 3.9560e+02, 1.2430e+01],
                     [1.4455e-01, 1.2500e+01, 7.8700e+00, 0.0000e+00, 5.2400e-01,
                      6.1720e+00, 9.6100e+01, 5.9505e+00, 5.0000e+00, 3.1100e+02,
                      1.5200e+01, 3.9690e+02, 1.9150e+01],
                     [2.1124e-01, 1.2500e+01, 7.8700e+00, 0.0000e+00, 5.2400e-01,
                      5.6310e+00, 1.0000e+02, 6.0821e+00, 5.0000e+00, 3.1100e+02,
                      1.5200e+01, 3.8663e+02, 2.9930e+01],
                     [1.7004e-01, 1.2500e+01, 7.8700e+00, 0.0000e+00, 5.2400e-01,
                      6.0040e+00, 8.5900e+01, 6.5921e+00, 5.0000e+00, 3.1100e+02,
                      1.5200e+01, 3.8671e+02, 1.7100e+01]])

    y = numpy.array([24., 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9])

    Xsp = scipy.sparse.csr_matrix(X)

    lr = LinearLearner(lambda_=0) #init model (samo nastavi lambdo)
    linear = lr(Xsp, y) #nauci model

    for a in X:
        print(linear(a)) #napovej razred za primer z modelom
