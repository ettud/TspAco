import unittest
from main import *
import datetime


class Test(unittest.TestCase):
    alpha = 0.3
    beta = 0.75
    p = 0.75
    t = 80
    ants_number = 10

    def test_solve_tsp_with_aco(self):
        graph = np.array([  [0, 2, 30, 9, 1],
                            [4, 0, 47, 7, 7],
                            [31, 33, 0, 33, 36],
                            [20, 13, 16, 0, 28],
                            [9, 36, 22, 22, 0]  ])
        result = solve_tsp_with_aco(graph, self.ants_number, self.alpha, self.beta,self. p, self.t)
        self.assertEqual(70, result[1])

    def test_for_time(self):
        graph = np.random.randint(1, 50, size=(5, 5))
        np.fill_diagonal(graph, 0)
        time1 = datetime.datetime.now()
        result = solve_tsp_with_aco(graph, self.ants_number, self.alpha, self.beta, self.p, self.t)
        time2 = datetime.datetime.now()
        self.assert_(time2-time1 < datetime.timedelta(seconds=30))


if __name__ == "__main__":
    unittest.main()
