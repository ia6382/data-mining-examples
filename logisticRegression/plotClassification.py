from matplotlib import pyplot
import numpy as np

from logisticRegression import load, LogRegLearner, optimalLambda, CA, R2, test_learning, test_cv


def draw_decision(X, y, classifier, at1, at2, grid=50):

    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    pyplot.figure(figsize=(8,8))

    for c,(x,y) in zip(y,points):
        pyplot.text(x,y,str(c), ha="center", va="center")
        pyplot.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi,x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pyplot.imshow(prob, extent=(minx,maxx,maxy,miny), cmap="seismic")

    pyplot.xlim(minx, maxx)
    pyplot.ylim(miny, maxy)
    pyplot.xlabel(at1)
    pyplot.ylabel(at2)

    pyplot.show()


if __name__ == "__main__":
    X,y = load('reg.data')
    #lambdas = np.arange(0, 1, 0.1)
    lambdas = [0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 10, 100]

    #najdi optimalno lambdo
    optl = optimalLambda(test_cv, R2, X, y, lambdas)
    print("Najboljsa lambda: "+str(optl))
    
    #nariši klasifikacijo primerov
    learner = LogRegLearner(lambda_=optl)
    model = learner(X,y)
    draw_decision(X, y, model, 0, 1)

    #naredi latex tabelo točnosti modela za različne vrednosti lambd
    print("\n\\begin{table}[htbp]\n\t\\begin{center}\n\t\t\\begin{tabular}{c| c| c| c| c}\n\t\t\t\\hline & \multicolumn{2}{c|}{learning} & \multicolumn{2}{c}{cv}\\\ \n\t\t\t$\lambda$ & CA & R2 & CA & R2 \\\ \n\t\t\t\\hline")
    for l in lambdas:
        learner = LogRegLearner(lambda_=l)
        predCv = test_cv(learner, X, y)
        predLr = test_learning(learner, X, y)

        print("\t\t\t %.3f & %.3f & %.3f & %.3f & %.3f \\\ " %(l, CA(y, predLr), R2(y, predLr), CA(y, predCv), R2(y, predCv)))
    print("\t\t\t\\hline\n\t\t\\end{tabular}\n\t\t\\caption{Točnosti modela pri različnih stapnjah regularizacije ($\lambda$). }\n\t\t\\label{tabela}\n\t\\end{center}\n\\end{table}\n")
    

    
