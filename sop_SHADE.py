import numpy as np

from scipy.stats import cauchy
# from problems.bounds_cop import Rastrigin
import time
from problems.bounds_cop import CEC2022

class SHADE:
    """Success-History based Adaptive Differential Evolution (SHADE).
    """
    
    def __init__(self, problem, popsize, h, maxfes):
        
        self.popsize = popsize
        self.problem = problem
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.max_iter = int(maxfes / self.popsize)

        self.init_f_mean = 0.5 
        self.init_cr_mean = 0.5 
        self.h = h

        
        self.m_cr = np.ones(self.h)*self.init_cr_mean # means of normal distribution
        self.m_f = np.ones(self.h)*self.init_f_mean  # medians of Cauchy distribution
        

        self._k = 0  # index to update
        self.p_min = 2.0/self.popsize

        self.best_so_far_y, self.best_so_far_x = np.Inf, None

        self.n_fes = 0
        self.current_generation = 0

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
        a = np.empty((0, self.dim))  # set of archived inferior solutions
        return x, y, a


    def bound(self, x=None, xx=None):
        for k in range(self.popsize):
            idx = np.array(x[k] < self.lb)
            if idx.any():
                x[k][idx] = (self.lb + xx[k])[idx]/2.0
            idx = np.array(x[k] > self.ub)
            if idx.any():
                x[k][idx] = (self.ub + xx[k])[idx]/2.0
        return x

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.popsize, self.dim))  # mutated population
        f_mu = np.empty((self.popsize,))  # mutated mutation factors
        x_un = np.vstack((np.copy(x), a))  # union of population x and archive a
        r = np.random.choice(self.h, (self.popsize,))
        order = np.argsort(y)[:]
        p = (0.2 - self.p_min)*np.random.random((self.popsize,)) + self.p_min
        idx = [order[np.random.choice(int(i))] for i in np.ceil(p*self.popsize)]
        for k in range(self.popsize):
            f_mu[k] = cauchy.rvs(loc=self.m_f[r[k]], scale=0.1)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.m_f[r[k]], scale=0.1)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = np.random.choice([i for i in range(self.popsize) if i != k])
            r2 = np.random.choice([i for i in range(len(x_un)) if i != k and i != r1])
            x_mu[k] = x[k] + f_mu[k]*(x[idx[k]] - x[k]) + f_mu[k]*(x[r1] - x_un[r2])
        return x_mu, f_mu, r
    
    def crossover(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x)
        p_cr = np.empty((self.popsize,))  # crossover probabilities
        for k in range(self.popsize):
            p_cr[k] = np.random.normal(self.m_cr[r[k]], 0.1)
            p_cr[k] = np.clip(p_cr[k], 0.0, 1.0) 
            i_rand = np.random.randint(self.dim)
            for i in range(self.dim):
                if (i == i_rand) or (np.random.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr
    
    def _evaluate_fitness(self, x):
        y = self.problem.objective_function(x)
        self.n_fes += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        return float(y)
    
    def select(self, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        # set successful muta        # self.p = 0.05
        # self.c = 0.1tion factors, crossover probabilities, fitness differences
        f, p, d = np.empty((0,)), np.empty((0,)), np.empty((0,))
        for k in range(self.popsize):
            
            yy = self._evaluate_fitness(x_cr[k])
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive of inferior solutions
                f = np.hstack((f, f_mu[k]))  # archive of successful mutation factors
                p = np.hstack((p, p_cr[k]))  # archive of successful crossover probabilities
                d = np.hstack((d, y[k] - yy))  # archive of successful fitness differences
                x[k], y[k] = x_cr[k], yy

        if (len(p) != 0) and (len(f) != 0):
            w = d/np.sum(d)
            self.m_cr[self._k] = np.sum(w*p)  # for normal distribution
            self.m_f[self._k] = np.sum(w*np.power(f, 2))/np.sum(w*f)  # for Cauchy distribution
            self._k = (self._k + 1) % self.h
        return x, y, a
    
    def iterate(self, x=None, y=None, a=None):
        x_mu, f_mu, r = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x, r)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(x, y, x_cr, a, f_mu, p_cr)
        if len(a) > self.popsize:  # randomly remove solutions to keep archive size fixed
            a = np.delete(a, np.random.choice(len(a), (len(a) - self.popsize,), False), 0)
        return x, y, a
    
    def optimize(self):
        x, y, a = self.initialize()
        for i in range(1, self.max_iter+1):
            self.current_generation += 1
            # if i % 10 == 0:
            #     info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            #     print(info.format(self.current_generation, self.best_so_far_y, np.min(y), self.n_fes))
            x, y, a = self.iterate(x, y, a)
        return self.best_so_far_x, self.best_so_far_y
    

if __name__ == "__main__":
    f_lst = [
         "F2", "F4",  "F6", "F7", "F8", "F9", "F12"
        
    ]

    fb_lst = [
        400, 800, 1800, 2000, 2200, 2300, 2700
    ]

    M = 10 # 独立运行次数

    ex_time = 0

    for k,fname in enumerate(f_lst):
        problem = CEC2022(fname, dim=10)
       
        err_lst = []
        start = time.time()
        for i in range(M):
            de = SHADE(problem, 100, 10, 1e5)
            bestx, besty = de.optimize()
            err_lst.append(besty - fb_lst[k])
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"SHADE 算法在问题{fname}上10次独立实验目标值结果: mean(std)={np.mean(err_lst):.2e}({np.std(err_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")


    


