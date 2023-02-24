import unittest
import numpy as np

from solution import power_iteration, power_iteration_two_components, \
    project_to_eigenvectors, explained_variance_ratio


DATA = np.array([[22.0, 81.0, 32.0, 39.0, 21.0, 37.0, 46.0, 36.0, 99.0],
                 [91.0, 95.0, 65.0, 96.0, 89.0, 39.0, 11.0, 22.0, 29.0],
                 [51.0, 89.0, 21.0, 39.0, 100.0, 59.0, 100.0, 89.0, 27.0],
                 [9.0, 80.0, 18.0, 34.0, 61.0, 100.0, 90.0, 92.0, 8.0],
                 [93.0, 99.0, 39.0, 100.0, 12.0, 47.0, 17.0, 12.0, 63.0],
                 [49.0, 83.0, 17.0, 33.0, 92.0, 30.0, 98.0, 91.0, 73.0],
                 [91.0, 99.0, 97.0, 89.0, 49.0, 96.0, 81.0, 94.0, 69.0],
                 [12.0, 69.0, 32.0, 14.0, 34.0, 12.0, 33.0, 48.0, 96.0],
                 [91.0, 80.0, 20.0, 10.0, 82.0, 93.0, 87.0, 91.0, 22.0],
                 [39.0, 100.0, 19.0, 29.0, 99.0, 31.0, 77.0, 79.0, 23.0],
                 [20.0, 91.0, 10.0, 15.0, 71.0, 99.0, 78.0, 93.0, 12.0],
                 [90.0, 60.0, 45.0, 34.0, 45.0, 20.0, 15.0, 5.0, 100.0],
                 [100.0, 98.0, 97.0, 89.0, 32.0, 72.0, 22.0, 13.0, 37.0],
                 [14.0, 4.0, 15.0, 27.0, 61.0, 42.0, 51.0, 52.0, 39.0],
                 [9.0, 22.0, 8.0, 7.0, 100.0, 11.0, 92.0, 96.0, 29.0],
                 [85.0, 90.0, 100.0, 99.0, 45.0, 38.0, 92.0, 67.0, 21.0]])

EVECS = np.array([[0.42089683, 0.17659843, 0.38473494, 0.43182306, -0.3039823,
                   -0.03955052, -0.37855211, -0.42617528, 0.19562766],
                  [-0.32014675, -0.32874635, -0.26358331, -0.28729244, -0.1635365,
                   -0.45716676, -0.33710698, -0.32793313, 0.42484339]])

EVALS = np.array([4303.49765617, 2425.69800137])


PROJECTION = np.array([[ 20.69548364,  62.75243571],
                       [ 74.29185404, -14.39810189],
                       [-51.91681331, -35.41238311],
                       [-65.47337793, -34.83095806]])


def maybe_negate(vec, comparison):
    if np.abs(np.sum(vec - comparison)) < np.abs(np.sum(-vec - comparison)):
        return vec
    else:
        return -vec


class PowerIterationPCATest(unittest.TestCase):

    def test_power_iteration_single(self):
        vec, val = power_iteration(DATA)

        np.testing.assert_almost_equal(val, EVALS[0], decimal=3)

        vec = maybe_negate(vec, EVECS[0])
        np.testing.assert_almost_equal(vec, EVECS[0], decimal=3)

    def test_power_iteration_two_components(self):
        vecs, vals = power_iteration_two_components(DATA)

        np.testing.assert_almost_equal(vals, EVALS, decimal=3)

        for vec, correct in zip(vecs, EVECS):
            vec = maybe_negate(vec, correct)
            np.testing.assert_almost_equal(vec, correct, decimal=3)

    def test_project_to_eigenvectors(self):
        projection = project_to_eigenvectors(DATA, EVECS)
        for i in range(2):
            comp = projection[:, i][:4]
            correct = PROJECTION[:, i]
            np.testing.assert_almost_equal(comp, correct, decimal=3)

    def test_explained_variance(self):
        e = explained_variance_ratio(DATA, EVECS, EVALS)
        np.testing.assert_almost_equal(e, 0.7131618003969974, decimal=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
