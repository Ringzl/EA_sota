import math
import numpy as np

from problems.bounds_cop import Rastrigin

class RCGA:

    def __init__(self, problem, popsize, maxfes):
        self.popsize = popsize
        self.problem = problem
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.max_iter = int(maxfes / self.popsize)

        self.best_so_far_y, self.best_so_far_x = np.Inf, None
        self.n_fes = 0
        self.current_generation = 0

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

    
    def initialize(self):
        # x = np.random.uniform(self.lb, self.ub, size=(self.popsize, self.dim))  # population
        x = np.empty((self.popsize, self.dim))
        for j in range(self.dim):
            for i in range(self.popsize):
                x[i, j] = self.lb[j] + np.random.uniform(i / self.popsize * (self.ub[j] - self.lb[j]), (i + 1) / self.popsize * (self.ub[j] - self.lb[j]))
            np.random.shuffle(x[:, j])
        
        y = np.empty((self.popsize,))  # fitness
        for i in range(self.popsize):
            y[i] = self._evaluate_fitness(x[i])
        return x, y
    
    def _evaluate_fitness(self, x):
        y = self.problem.objective_function(x)
        self.n_fes += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        return float(y)

    def RCGA(self, X):
        matingpool = np.arange(X.shape[0])
        np.random.shuffle(matingpool)
        X = X[matingpool]

        X1 = X[0: math.floor(X.shape[0] / 2), :]
        X2 = X[math.floor(X.shape[0] / 2): math.floor(X.shape[0] / 2) * 2, :]
        N = X1.shape[0]
        D = X1.shape[1]

        beta = np.zeros((N, D))
        mu = np.random.random((N, D))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (self.disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (self.disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        beta[np.random.random((N, D)) < 0.5] = 1
        beta[np.tile(np.random.random((N, 1)) > self.proC, (1, D))] = 1
        Offspring = np.vstack(
            (
                (X1 + X2) / 2 + beta * (X1 - X2) / 2,
                (X1 + X2) / 2 - beta * (X1 - X2) / 2,
            )
        )
        Lower = np.tile(self.lb[self.dim - X.shape[1]:], (2 * N, 1))
        Upper = np.tile(self.ub[self.dim - X.shape[1]:], (2 * N, 1))
        Site = np.random.random((2 * N, D)) < self.proM / D
        mu = np.random.random((2 * N, D))
        temp = np.logical_and(Site, mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
                (
                        2 * mu[temp]
                        + (1 - 2 * mu[temp])
                        * (  # noqa
                                1
                                - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])  # noqa
                        )
                        ** (self.disM + 1)
                )
                ** (1 / (self.disM + 1))
                - 1
        )  # noqa
        temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
                1
                - (
                        2 * (1 - mu[temp])
                        + 2
                        * (mu[temp] - 0.5)
                        * (  # noqa
                                1
                                - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])  # noqa
                        )
                        ** (self.disM + 1)
                )
                ** (1 / (self.disM + 1))
        )
        Offspring = np.clip(Offspring, self.lb[self.dim - X.shape[1]:], self.ub[self.dim - X.shape[1]:])
        return Offspring
    
    def optimize(self):
        x, y = self.initialize()

        for i in range(1, self.max_iter+1):

            if i % 10 == 0:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self.current_generation, self.best_so_far_y, np.min(y), self.n_fes))
    
            ox = self.RCGA(x)
            oy = np.empty(ox.shape[0])
            for i in range(x.shape[0]):
                oy[i] = self._evaluate_fitness(ox[i])

            cx = np.vstack([x, ox])
            cy = np.append(y,oy)

            next = np.argsort(cy)[:self.popsize]

            x = cx[next, :]
            y = cy[next]
        
        return self.best_so_far_x, self.best_so_far_y

if __name__ == "__main__":
    problem = Rastrigin()
    shade = RCGA(problem, 100, 1e4)
    bestx, besty = shade.optimize()

    print(f"Best x: {bestx}, Best y: {besty}")
