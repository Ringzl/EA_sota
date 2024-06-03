import time
import random
import numpy as np

from problems.scop import CEC2017
from tqdm import tqdm

def nahvyb_expt(N, k, expt=None):
    opora = list(range(N))  # Create a list from 0 to N-1
    if expt is not None:  # Check if expt is provided and remove it
        try:
            opora.remove(expt)  # Remove the specified exempted value
        except ValueError:
            pass  # If expt is not in the list, do nothing
    vyb = [0] * k  # Initialize a list of zeros with length k
    for i in range(k):
        index = random.randint(0, len(opora) - 1)  # Generate a random index
        vyb[i] = opora[index]  # Take the element at that index
        opora.pop(index)  # Remove the selected element from the list
    return vyb

def zrcad(y, a, b):
    zrc = np.where((y < a) | (y > b))[0]
    for i in zrc:
        if y[i] > b[i]:
            y[i] = 2 * b[i] - y[i]
        elif y[i] < a[i]:
            y[i] = 2 * a[i] - y[i]
    return y

class DeCODE:
    def __init__(self, prob, popsize, maxfes):

        self.dim = prob.dim
        self.popsize = popsize
        self.prob = prob
        self.maxfes = maxfes

        self.alpha = 0.75
        self.beta = 6
        self.gama = 30
        self.mu = 1e-8
        self.P = 0.85

        self.F_pool = [0.6, 0.8, 1.0]
        self.CR_pool = [0.1, 0.2, 1.0]

        self.FEs = 0
        self.weights = None

        # archive
        self.arch_x = None
        self.arch_obj = np.empty(self.popsize)
        self.arch_cv = np.empty(self.popsize)
        
        # population
        self.x = None
        self.cv = np.empty(self.popsize)
        self.obj = np.empty(self.popsize)

        self.best_so_far_x, self.best_so_far_y = None, float('inf')


    def initialize(self):
        self.x = np.random.uniform(self.prob.lb, self.prob.ub, size=(self.popsize, self.dim))  # population
        for i in range(self.popsize):
            obj, cv_total = self._evaluate_fitness(self.x[i])
            self.obj[i] = obj
            self.cv[i] = cv_total  # 总约束违反

    def Generation(self):
        normal_obj = (self.obj - np.min(self.obj))/(np.std(self.obj) + 2e-16)
        normal_cv  = (self.cv - np.min(self.cv))/(np.std(self.cv) + 2e-16)

        # 
        offx = np.empty_like(self.x)
        for i in range(self.popsize):
            offx[i] = self.x[i].copy()
            A = nahvyb_expt(self.popsize, 3, i)
            r1, r2, r3 = A[0], A[1], A[2]
            F = self.F_pool[np.random.randint(len(self.F_pool))]
            CR = self.CR_pool[np.random.randint(len(self.CR_pool))]

            if random.random() < self.FEs/self.maxfes:
                # rand-to-best
                fit = self.weights[i] * normal_obj + (1-self.weights[i]) * normal_cv
                best = np.argmin(fit)

                v = self.x[r1] + F * (self.x[best] - self.x[r1]) + F * (self.x[r2] - self.x[r3])
                
                change = np.where(np.random.rand(self.dim) < CR)[0]  #
                offx[i][change] = v[change]

            else:
                # current-to-rand
                offx[i] = self.x[i] + random.random()*(self.x[r1] - self.x[i]) + F * (self.x[r2] - self.x[r3])

            offx[i] = zrcad(offx[i], self.prob.lb, self.prob.ub)
        
        return offx

    def slection(self, offx, offobj, offcv):
        objs = np.append(self.obj, offobj)
        cvs = np.append(self.cv, offcv)

        normal_objs = (objs-np.min(objs))/(np.std(objs) + 2e-16)
        normal_cvs = (cvs - np.min(cvs))/(np.std(cvs) + 2e-16)

        normal_pop_obj, normal_off_obj = normal_objs[:len(self.obj)], normal_objs[len(self.obj):]
        normal_pop_cvs, normal_off_cvs = normal_cvs[:len(self.cv)], normal_cvs[len(self.cv):]

        pop_fit = self.weights * normal_pop_obj + (1-self.weights) * normal_pop_cvs
        off_fit = self.weights * normal_off_obj + (1-self.weights) * normal_off_cvs

        replace = np.where(pop_fit > off_fit)[0]

        self.x[replace, :] = offx[replace, :]
        self.obj[replace] = offobj[replace]
        self.cv[replace] = offcv[replace]

    def Selection_Tournament(self, offx, offobj, offcv, Ep=0):
        replace_cv = (self.cv > offcv) & (self.cv > Ep) & (offcv > Ep) # 约束违反小
        equal_cv = (self.cv <= Ep) & (offcv <= Ep)  # 都可行
        replace_f = self.obj > offobj
        replace = (equal_cv & replace_f) | replace_cv

        self.arch_x[replace, :] = offx[replace,:]
        self.arch_obj[replace] = offobj[replace]
        self.arch_cv[replace] = offcv[replace]
    

    def _evaluate_fitness(self, x):
        obj, cv = self.prob.evaluate(x)
        cv_total = np.sum(cv)  # 总约束违反
        self.FEs += 1

        # update best-so-far solution (x) and fitness (y)
        if cv_total == 0 and obj < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = x, obj
        return obj, cv_total
    
    def run(self):

        self.initialize()
        self.arch_x = self.x.copy()
        self.arch_obj = self.obj.copy()
        self.arch_cv = self.cv.copy()

        Ep0 = min(10**self.dim / 2, max(self.cv))
        cp = (-np.log(Ep0) - self.beta) / np.log(1 - self.P)
        rate = 0 # X
        pmax = 1 # pmax

        while self.FEs < self.maxfes:
            if rate < self.P:
                Ep =  Ep0 * (1 - rate)**cp
            else:
                Ep = 0
            rate += self.popsize/(self.maxfes)

            if len(np.where(self.cv == 0)[0]) > self.P * self.popsize:
                Ep = 0

            rand_idx = np.random.permutation(list(range(self.popsize)))
            self.x = self.x[rand_idx]
            self.obj = self.obj[rand_idx]
            self.cv = self.cv[rand_idx]

            self.arch_x = self.arch_x[rand_idx]
            self.arch_obj = self.obj[rand_idx]
            self.arch_cv = self.cv[rand_idx]

            if len(np.where(self.cv < Ep)[0]) == 0:
                pmax = 1e-18
            pr = max(1e-18, pmax / (1 + np.exp(self.gama * (self.FEs/self.maxfes - self.alpha))))

            # diversity restart
            if np.std(self.cv) < self.mu and len(np.where(self.cv == 0)[0]) == 0:
                self.initialize()
            
            self.weights = np.linspace(0, pr, self.popsize)
            rand_idx = np.random.permutation(list(range(len(self.weights))))
            self.weights = self.weights[rand_idx]
            # Generation
            offx = self.Generation()
            offobj = np.empty(offx.shape[0])
            offcv = np.empty(offx.shape[0])
            for i in range(offx.shape[0]):
                obj, cv_total = self._evaluate_fitness(offx[i])
                offobj[i] = obj
                offcv[i] = cv_total

            # if self.FEs % 10 == 0:
            #     info = '  * Generation {:d}: best_so_far_y {:7.5e} & Evaluations {:d}'
            #     print(info.format(self.FEs//self.popsize, self.best_so_far_y, self.FEs))
            
            # selection
            self.slection(offx, offobj, offcv)
            self.Selection_Tournament(offx, offobj, offcv)
        
        return self.best_so_far_x, self.best_so_far_y


if __name__ == "__main__":
    
    # prob = CEC2017("C05", 10)

    # decode = DeCODE(prob, 100, 1e5)

    # bestx, besty = decode.run()
    
    # print(prob.evaluate(bestx))


    f_lst = [
         "C01", "C02",  "C03", "C04", "C05"
    ]

    M = 10 # 独立运行次数

    ex_time = 0

    for k,fname in enumerate(f_lst):
        problem = CEC2017(fname, dim=10)
       
        err_lst = []
        start = time.time()
        for i in tqdm(range(M)):
            de = DeCODE(problem, 100, 1e5)
            bestx, besty = de.run()
            if bestx is not None:
                err_lst.append(besty)
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"DeCODE 算法在问题{fname}上10次独立实验目标值结果: 可行解比例{len(err_lst)/10:.2%} mean(std)={np.mean(err_lst):.2e}({np.std(err_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")



