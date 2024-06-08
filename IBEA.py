import math
import numpy as np

import time
from pymoo.problems.multi import ZDT1
from pymoo.problems.many.dtlz import DTLZ2, DTLZ3
from pymoo.problems.many.wfg import WFG1, WFG4
from pymoo.indicators.hv import HV

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pymoo.indicators.igd import IGD
# from pymoo.indicators.igd_plus import IGDPlus

import warnings
warnings.filterwarnings("ignore")
from pymoo.config import Config
Config.warnings['not_compiled'] = False

class Pop(object):
    def __init__(self, X):
        self.X = X
        self.ObjV = None

    def __add__(self, other):
        X = np.vstack([self.X, other.X])
        ObjV = np.vstack([self.ObjV, other.ObjV])
        newp = Pop(X)
        newp.ObjV = ObjV
        return newp


class IBEA(object):
    def __init__(self, prob, popsize, MaxFEs):

        self.MaxFEs = MaxFEs
        self.popsize = popsize

        self.dim = prob.n_var
        self.xmin = prob.bounds()[0]
        self.xmax = prob.bounds()[1]
        self.func = prob.evaluate

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

        self.kappa = 0.05
        self.pop = None
        self.FEs = 0

    def initialization(self):
        # 种群初始化
        X = np.zeros((self.popsize, self.dim))
        area = self.xmax - self.xmin
        for j in range(self.dim):
            for i in range(self.popsize):
                X[i, j] = self.xmin[j] + np.random.uniform(i / self.popsize * area[j],
                                                           (i + 1) / self.popsize * area[j])
            np.random.shuffle(X[:, j])

        self.pop = Pop(X)
        self.pop.ObjV = self.func(X)
        self.FEs = self.popsize

    def reproduction(self, X):
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
        Lower = np.tile(self.xmin, (2 * N, 1))
        Upper = np.tile(self.xmax, (2 * N, 1))
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
        Offspring = np.clip(Offspring, self.xmin, self.xmax)
        return Offspring

    # 计算I指标作为适应度值
    def CalFitness(self, pop):
        N = pop.ObjV.shape[0]
        # 归一化
        fmin = np.min(pop.ObjV, axis = 0)
        fmax = np.max(pop.ObjV, axis = 0)
        objv = (pop.ObjV - fmin)/(fmax - fmin)
        I = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                I[i,j] = np.max(objv[i] - objv[j], axis=0)
        C = np.max(np.abs(I), axis=0)
        Fitness = np.sum(-np.exp(-I/C/self.kappa), axis = 0) + 1

        return Fitness, I, C


    def EnviromentalSelection(self, pop):
        Next = list(range(pop.ObjV.shape[0]))
        Fitness, I, C  = self.CalFitness(pop)

        while len(Next) > self.popsize:
            x = np.argmin(Fitness[Next])
            Fitness = Fitness + np.exp(-I[Next[x]]/C[Next[x]]/self.kappa)
            Next.remove(Next[x])
        return Next


    # 进行K-锦标赛选择， args 输入objv，或rank
    def tournament_selection(self, K, N, *args):
        for i in range(len(args)):
            args[i].reshape(1, -1)
        fitness = np.vstack(args)
        _, rank = np.unique(fitness, return_inverse=True, axis=1)
        parents = np.random.randint(low=0, high=len(rank), size=(N, K))
        best = np.argmin(rank[parents], axis=1)
        index = parents[np.arange(N), best]
        return index

    def run(self):
        self.initialization()

        while(self.FEs < self.MaxFEs):
            
            popF,_,_ = self.CalFitness(self.pop)
            matingpool = self.tournament_selection(2, self.popsize, -popF)
            newX = self.reproduction(self.pop.X[matingpool])
            newf = self.func(newX)
            self.FEs += newf.shape[0]

            allX = np.vstack([self.pop.X, newX])
            cpop = Pop(allX)
            cpop.ObjV = np.vstack([self.pop.ObjV, newf])

            Next = self.EnviromentalSelection(cpop)

            # 更新种群
            self.pop.X = cpop.X[Next]
            self.pop.ObjV = cpop.ObjV[Next]

        return self.pop.X, self.pop.ObjV


def plot_NDS(PF, F):
    fig = plt.figure()
    if PF.shape[1] == 2:
        plt.plot(PF[:, 0], PF[:, 1], label = "Pareto Front")
        plt.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolor="red", label = "Approximate PF")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.legend()
    elif PF.shape[1] == 3:
        x1, y1, z1 = PF[:, 0], PF[:, 1], PF[:, 2]
        x2, y2, z2 = F[:,0], F[:,1], F[:,2]
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.view_init(elev=30, azim=45)
        ax.scatter(x1, y1, z1, marker="s", label = "Pareto Front")
        ax.scatter(x2, y2, z2, facecolor="none", edgecolor="red", label = "Approximate PF")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        plt.legend()
    plt.show()

if __name__ == "__main__":
    # problem = ZDT1(n_var=10)
    # ibea = IBEA(problem, 100, 1e5)
    # igd = IGD(problem.pareto_front())
    # X, F = ibea.run()
    # print(igd(F))
    # plot_NDS(problem.pareto_front(), F)
    p_dct = {
        'WFG1': WFG1,   
        'WFG4': WFG4,
        'DTLZ2': DTLZ2,
        'DTLZ3': DTLZ3
    }

    M = 10 # 独立运行次数
    n_var = 10
    n_obj = 3

    refpoint = np.array([2 * i + 1 for i in range(1, n_obj+1)])
    ind = HV(ref_point=refpoint)


    
    ex_time = 0
    for k, fname in enumerate(p_dct):

        problem = p_dct[fname](n_var=10, n_obj=3)
        hv_lst = []
        igd_lst = []
        start = time.time()
        for i in range(M):
            alg = IBEA(problem, 100, 1e5)
            X, F = alg.run()
            hv = ind(F)
            igd = IGD(problem.pareto_front())
            igd_lst.append(igd(F))
            hv_lst.append(hv/np.prod(refpoint))
        
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"IBEA 算法在问题{fname}上10次独立实验目标值HV结果: mean(std)={np.mean(hv_lst):.2e}({np.std(hv_lst):.2e})")
        print(f"IBEA 算法在问题{fname}上10次独立实验目标值IGD结果: mean(std)={np.mean(igd_lst):.2e}({np.std(igd_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")