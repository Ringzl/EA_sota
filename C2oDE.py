
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

def minFP(objs, cvs):

    cvs[cvs <= 0] = 0
    minCV = np.min(cvs)
    idx_min_cv = np.where(cvs == minCV)[0]
    obj_temp = objs[idx_min_cv]

    idx_temp = np.argmin(obj_temp)
    minObj = obj_temp[idx_temp]
    min_idx = idx_min_cv[idx_temp]

    return minObj, minCV, min_idx


class C2oDE:
    def __init__(self, prob, popsize, maxfes):

        self.dim = prob.dim
        self.popsize = popsize
        self.prob = prob
        self.maxfes = maxfes
        
        self.beta = 6
        self.mu = 1e-8
        self.P = 0.5

        self.F_pool = [0.6, 0.8, 1.0]
        self.CR_pool = [0.1, 0.2, 1.0]

        self.FEs = 0

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
        offx = np.empty((3*self.popsize, self.dim))
        for i in range(self.popsize):
            j = i*3
            offx[j] = self.x[i].copy()
            offx[j+1] = self.x[i].copy()
            offx[j+2] = self.x[i].copy()

            # current-to-best
            A = nahvyb_expt(self.popsize, 2, i)
            r1, r2 = A[0], A[1]
            F = self.F_pool[np.random.randint(len(self.F_pool))]
            CR = self.CR_pool[np.random.randint(len(self.CR_pool))]
            best = np.argmin(self.obj)
            v = self.x[i] + F * (self.x[best] - self.x[i]) + F * (self.x[r1] - self.x[r2])
            change = np.where(np.random.rand(self.dim) < CR)[0]  #
            offx[j][change] = v[change]

            # rand-to-best-modified
            A = nahvyb_expt(self.popsize, 4, i)
            r1, r2, r3, r4 = A[0], A[1], A[2], A[3]
            _, _, best = minFP(self.obj, self.cv)
            F = self.F_pool[np.random.randint(len(self.F_pool))]
            CR = self.CR_pool[np.random.randint(len(self.CR_pool))]
            v = self.x[r1] + F * (self.x[best] - self.x[r2]) + F * (self.x[r3] - self.x[r4])
            change = np.where(np.random.rand(self.dim) < CR)[0]  #
            offx[j+1][change] = v[change]

            # current-to-rand
            A = nahvyb_expt(self.popsize, 3, i)
            r1, r2, r3 = A[0], A[1], A[2]
            F = self.F_pool[np.random.randint(len(self.F_pool))]
            offx[j+2] = self.x[i] + random.random() * (self.x[r1] - self.x[i]) + F * (self.x[r2] - self.x[r3])

            for t in range(j, j+3):
                offx[t] = zrcad(offx[t], self.prob.lb, self.prob.ub)

        return offx

    def Selection_Tournament(self, offx, offobj, offcv, Ep=0):
        replace_cv = (self.cv > offcv) & (self.cv > Ep) & (offcv > Ep) # 约束违反小
        equal_cv = (self.cv <= Ep) & (offcv <= Ep)  # 都可行
        replace_f = self.obj > offobj
        replace = (equal_cv & replace_f) | replace_cv

        self.x[replace, :] = offx[replace,:]
        self.obj[replace] = offobj[replace]
        self.cv[replace] = offcv[replace]
            
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
        Ep0 = np.max(self.cv)
        rate = 0
        cp = (-np.log(Ep0) - self.beta) / np.log(1 - self.P)

        while self.FEs < self.maxfes:
            if rate < self.P:
                Ep =  Ep0 * (1 - rate)**cp
            else:
                Ep = 0
            rate += (self.popsize * 3)/self.maxfes

            # diversity restart
            if np.std(self.cv) < self.mu and len(np.where(self.cv == 0)[0]) == 0:
                self.initialize()
            
            # Generation
            offx_temp = self.Generation()
            offobj_temp = np.empty(offx_temp.shape[0])
            offcv_temp = np.empty(offx_temp.shape[0])
            for i in range(offx_temp.shape[0]):
                obj, cv_total = self._evaluate_fitness(offx_temp[i])
                offobj_temp[i] = obj
                offcv_temp[i] = cv_total

            # if self.FEs % 10 == 0:
            #     info = '  * Generation {:d}: best_so_far_y {:7.5e} & Evaluations {:d}'
            #     print(info.format(self.FEs//self.popsize, self.best_so_far_y, self.FEs))

            # Pre Selection
            offx = np.empty_like(self.x)
            offobj = np.empty(offx.shape[0])
            offcv = np.empty(offx.shape[0])

            for i in range(self.popsize):
                idx = np.arange(3*i, 3*i + 3)
                _, _, best = minFP(offobj_temp[idx], offcv_temp[idx])
                
                offx[i] = offx_temp[idx[best]]
                offobj[i] = offobj_temp[idx[best]]
                offcv[i] = offcv_temp[idx[best]]

            # Selection
            self.Selection_Tournament(offx, offobj, offcv, Ep)

        return self.best_so_far_x, self.best_so_far_y


if __name__ == "__main__":
    
    # prob = CEC2017("C05", 10)

    # c2ode = C2oDE(prob, 100, 1e5)

    # bestx, besty = c2ode.run()
    
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
            de = C2oDE(problem, 100, 1e5)
            bestx, besty = de.run()
            if bestx is not None:
                err_lst.append(besty)
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"C2oDE 算法在问题{fname}上10次独立实验目标值结果: 可行解比例{len(err_lst)/10:.2%} mean(std)={np.mean(err_lst):.2e}({np.std(err_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")




                


    