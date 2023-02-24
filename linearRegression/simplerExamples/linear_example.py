import linear
import numpy

if __name__ == "__main__":

    X = numpy.array([[1,3],
                     [2,2],
                     [3,3]])

    y = numpy.array([10,11,12])

    lr = linear.LinearLearner(lambda_=1.)
    napovednik = lr(X,y)

    print "Koeficienti", napovednik.th #prvi je konstanten faktor

    nov_primer = numpy.array([2,11])
    print "Napoved", napovednik(nov_primer)
