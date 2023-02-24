#import sys
#print(sys.path)

import unittest
import numpy as np
from scipy.spatial import distance

from clustering import HierarchicalClustering


DATA = {'Albert': [22.0, 81.0, 32.0, 39.0, 21.0, 37.0, 46.0, 36.0, 99.0],
        'Branka': [91.0, 95.0, 65.0, 96.0, 89.0, 39.0, 11.0, 22.0, 29.0],
        'Cene': [51.0, 89.0, 21.0, 39.0, 100.0, 59.0, 100.0, 89.0, 27.0],
        'Dea': [9.0, 80.0, 18.0, 34.0, 61.0, 100.0, 90.0, 92.0, 8.0],
        'Edo': [93.0, 99.0, 39.0, 100.0, 12.0, 47.0, 17.0, 12.0, 63.0],
        'Franci': [49.0, 83.0, 17.0, 33.0, 92.0, 30.0, 98.0, 91.0, 73.0],
        'Helena': [91.0, 99.0, 97.0, 89.0, 49.0, 96.0, 81.0, 94.0, 69.0],
        'Ivan': [12.0, 69.0, 32.0, 14.0, 34.0, 12.0, 33.0, 48.0, 96.0],
        'Jana': [91.0, 80.0, 20.0, 10.0, 82.0, 93.0, 87.0, 91.0, 22.0],
        'Leon': [39.0, 100.0, 19.0, 29.0, 99.0, 31.0, 77.0, 79.0, 23.0],
        'Metka': [20.0, 91.0, 10.0, 15.0, 71.0, 99.0, 78.0, 93.0, 12.0],
        'Nika': [90.0, 60.0, 45.0, 34.0, 45.0, 20.0, 15.0, 5.0, 100.0],
        'Polona': [100.0, 98.0, 97.0, 89.0, 32.0, 72.0, 22.0, 13.0, 37.0],
        'Rajko': [14.0, 4.0, 15.0, 27.0, 61.0, 42.0, 51.0, 52.0, 39.0],
        'Stane': [9.0, 22.0, 8.0, 7.0, 100.0, 11.0, 92.0, 96.0, 29.0],
        'Zala': [85.0, 90.0, 100.0, 99.0, 45.0, 38.0, 92.0, 67.0, 21.0]}
CLUSTER_AVG_MAX = [[[['Rajko'], ['Stane']], [[['Franci'], [['Cene'], ['Leon']]],
                                             [['Jana'], [['Dea'], ['Metka']]]]],
                   [[['Nika'], [['Albert'], ['Ivan']]],
                    [[['Helena'], ['Zala']],
                     [['Branka'], [['Edo'], ['Polona']]]]]]
CLUSTER_MIN = [[[[['Dea'], ['Metka']], [['Jana'], [['Franci'],
                                                   [['Cene'], ['Leon']]]]],
                [['Rajko'], ['Stane']]], [[['Helena'], ['Zala']],
                                          [[['Branka'], [['Edo'], ['Polona']]],
                                           [['Nika'], [['Albert'], ['Ivan']]]]]]


def compare_trees(t1, t2):
    if len(t1) == 1 and len(t2) == 1:
        if t1[0].strip().lower() == t2[0].strip().lower():
            return True
        else:
            return False

    if len(t1) == 1 and len(t2) != 1:
        return False
    if len(t1) != 1 and len(t2) == 1:
        return False

    a0 = t1[0]
    b0 = t1[1]
    a1 = t2[0]
    b1 = t2[1]

    if compare_trees(a0, a1):
        left = True
    elif compare_trees(a0, b1):
        left = False
    else:
        return False

    if left:
        if compare_trees(b0, b1):
            return True
    else:
        if compare_trees(b0, a1):
            return True

    return False


class HierarchicalClusteringTest(unittest.TestCase):
    def setUp(self):
        self.data = DATA

    def test_row_distance(self):
        hc = HierarchicalClustering(self.data)

        available_answers = [175.803, 453.0]  # euclidean, Manhattan
        dist = hc.row_distance("Polona", "Rajko")

        equal = np.isclose(dist, available_answers, atol=1e-2)

        self.assertTrue(equal.any())

    def test_cluster_distance(self):
        hc = HierarchicalClustering(self.data)
        ca = [["Albert"], [["Branka"], ["Cene"]]]
        cb = [["Nika"], ["Polona"]]
        available_dists = [124.99, 165.86, 75.94]

        hc.row_distance = lambda a, b: distance.euclidean(
            self.data[a], self.data[b])

        equal = np.isclose(hc.cluster_distance(ca, cb),
                           available_dists, atol=1e-2)

        self.assertTrue(equal.any())

    def test_run(self):
        hc = HierarchicalClustering(self.data)

        hc.row_distance = lambda a, b: distance.euclidean(
            self.data[a], self.data[b])
        hc.run()

        self.assertTrue(compare_trees(hc.clusters, CLUSTER_AVG_MAX) or
                        compare_trees(hc.clusters, CLUSTER_MIN))


if __name__ == "__main__":
    unittest.main(verbosity=2)
