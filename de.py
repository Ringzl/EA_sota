import math
import numpy as np

from problems.bounds_cop import Rastrigin

class DE:

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

        self.F = 0.5
        self.CR = 0.8
    
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

    def mutation_crossover(self, x, y):
        muX = np.empty_like(x)
        b = np.argmin(y)
        
        for i in range(self.popsize):  # DE/rand/1
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = np.random.randint(0, self.popsize - 1)
                r2 = np.random.randint(0, self.popsize - 1)
                r3 = np.random.randint(0, self.popsize - 1)
            
            # DE/rand/1
            mutation = x[r1] + self.F * (x[r2] - x[r3])
            
            # DE/current-to-best/1
            # mutation = x[i] + self.F * (x[b]-x[i]) + self.F * (x[r1] - x[r2])

            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.lb[j] <= mutation[j] <= self.ub[j]:
                    muX[i, j] = mutation[j]
                else:
                    rand_value = self.lb[j] + np.random.random() * (self.ub[j] - self.lb[j])
                    muX[i, j] = rand_value

        trialX = np.empty_like(muX)
        for i in range(self.popsize):
            rj = np.random.randint(0, self.dim - 1)
            for j in range(self.dim):
                rf = np.random.random()
                if rf <= self.CR or rj == j:
                    trialX[i, j] = muX[i, j]
                else:
                    trialX[i, j] = x[i, j]
        return trialX
    


    def optimize(self):
        x, y = self.initialize()

        for i in range(1, self.max_iter+1):

            if i % 10 == 0:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self.current_generation, self.best_so_far_y, np.min(y), self.n_fes))
    
            ox = self.mutation_crossover(x, y)
            oy = np.empty(ox.shape[0])

            for i in range(x.shape[0]):
                oy[i] = self._evaluate_fitness(ox[i])


            # 环境选择
            for i in range(self.popsize):
                if oy[i] <= y[i]:
                    x[i] = ox[i]
                    y[i] = oy[i]
            


        
        return self.best_so_far_x, self.best_so_far_y

if __name__ == "__main__":
    problem = Rastrigin()

    shade = DE(problem, 100, 1e4)
    bestx, besty = shade.optimize()

    print(f"Best x: {bestx}, Best y: {besty}")
