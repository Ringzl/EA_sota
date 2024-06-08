import math
import numpy as np

from copy import deepcopy
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
        tmp = Pop(X)
        tmp.ObjV = ObjV
        return tmp


class TA2(object):
    def __init__(self, prob, popsize, MaxFEs):

        self.MaxFEs = MaxFEs
        self.popsize = popsize
        self.CAsize = popsize

        self.dim = prob.n_var
        self.xmin = prob.bounds()[0]
        self.xmax = prob.bounds()[1]
        self.func = prob.evaluate
        self.m = prob.n_obj
        self.p_norm = 1 / self.m


        self.CA = None
        self.DA = None
        self.pop = None
        self.FEs = 0

    # 判断支配关系：p支配q，返回1
    def is_dominated(self, p, q):
        if (np.any(p > q)):
            return 0
        elif (np.all(p == q)):
            return 0
        else:
            return 1

    def get_nds(self, ObjV):
        size = len(ObjV)
        F1 = []
        # 寻找pareto第一级个体
        for i in range(size):
            n_p = 0
            for j in range(size):
                if (i != j):
                    if (self.is_dominated(ObjV[j], ObjV[i])):
                        n_p += 1
            if (n_p == 0):
                F1.append(i)
        return F1

    def initlization(self):
        # 初始化种群
        # X = np.random.random((self.popsize, self.dim)) * (self.xmax - self.xmin) + self.xmin
        X = np.zeros((self.popsize, self.dim))
        area = self.xmax - self.xmin
        for j in range(self.dim):
            for i in range(self.popsize):
                X[i, j] = self.xmin[j] + np.random.uniform(i / self.popsize * area[j],
                                                           (i + 1) / self.popsize * area[j])
            np.random.shuffle(X[:, j])
        self.pop = Pop(X)
        self.pop.ObjV = self.func(X)
        self.FEs += self.popsize

        # 更新CA
        self.update_CA(self.pop)

        # 更新DA
        self.update_DA(self.pop)

    def update_CA(self, pop):
        # 先添加
        if self.CA is None or len(self.CA.ObjV) == 0:
            self.CA = Pop(pop.X)
            self.CA.ObjV = pop.ObjV
        else:
            self.CA.X = np.vstack([self.CA.X, pop.X])
            self.CA.ObjV = np.vstack([self.CA.ObjV, pop.ObjV])

        N = self.CA.ObjV.shape[0]

        if N <= self.CAsize:
            return

        # 适应度计算
        CAobj = (self.CA.ObjV - np.min(self.CA.ObjV, axis=0)) / (
                    np.max(self.CA.ObjV, axis=0) - np.min(self.CA.ObjV, axis=0))
        I = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                I[i, j] = np.max(CAobj[i] - CAobj[j])
        C = np.max(np.abs(I), axis=0)
        F = np.sum(-np.exp(-I / np.tile(C, (N, 1)) / 0.05), axis=0) + 1

        # 基于适应度删除部分解
        choose = list(range(N))
        while len(choose) > self.CAsize:
            x = np.argmin(F[choose])
            F = F + np.exp(-I[choose[x]] / C[choose[x]] / 0.05)
            choose.remove(choose[x])

        # 更新CA
        self.CA.X = self.CA.X[choose]
        self.CA.ObjV = self.CA.ObjV[choose]

    def update_DA(self, pop):
        # 先添加
        if self.DA is None or len(self.DA.ObjV) == 0:
            self.DA = Pop(pop.X)
            self.DA.ObjV = pop.ObjV
        else:
            self.DA.X = np.vstack([self.DA.X, pop.X])
            self.DA.ObjV = np.vstack([self.DA.ObjV, pop.ObjV])

        # 找非支配解,求DA
        ND = self.get_nds(self.DA.ObjV)
        N = len(ND)

        if N <= 1:
            return 

        self.DA.X = self.DA.X[ND]
        self.DA.ObjV = self.DA.ObjV[ND]

        if N <= self.popsize:
            return
        
        # 首先选出极点
        choose = np.full(N, False)
        extreme1 = np.argmin(self.DA.ObjV, axis=0)
        extreme2 = np.argmax(self.DA.ObjV, axis=0)
        choose[extreme1] = True
        choose[extreme2] = True

        # 删除或添加解
        if sum(choose) > self.popsize:

            # 随机删除一些解
            Choosed = np.where(choose)[0]
            k = np.random.randint(0, sum(choose), sum(choose) - self.popsize)
            choose[Choosed[k]] = False

        elif(sum(choose) < self.popsize):
            Distance = np.full((N,N), np.inf)
            for i in range(N-1):
                for j in range(i+1, N):
                    Distance[i,j] = np.linalg.norm(self.DA.ObjV[i]-self.DA.ObjV[j], self.p_norm)
                    Distance[j,i] = Distance[i,j]

            while(sum(choose) < self.popsize):
                Remain = np.where(1-choose)[0]
                x = np.argmax(np.min(Distance[(1-choose).astype(bool)][:, choose], axis = 1),axis=0)
                choose[Remain[x]] = True

        # 更新DA
        self.DA.X = self.DA.X[choose]
        self.DA.ObjV = self.DA.ObjV[choose]

    def matingSelection(self):
        CAP1 = np.random.randint(0,self.CA.ObjV.shape[0],math.ceil(self.popsize/2))
        CAP2 = np.random.randint(0,self.CA.ObjV.shape[0],math.ceil(self.popsize/2))

        Dominate = np.any(self.CA.ObjV[CAP1] < self.CA.ObjV[CAP2], axis = 1) ^ np.any(self.CA.ObjV[CAP1] > self.CA.ObjV[CAP2], axis = 1)

        cind1 = CAP1[Dominate==1]
        cx1 = self.CA.X[cind1]
        cy1 = self.CA.ObjV[cind1]

        cind2 = CAP2[Dominate!=1]
        cx2 = self.CA.X[cind2]
        cy2 = self.CA.ObjV[cind2]

        DA1 = np.random.randint(0, self.DA.ObjV.shape[0], math.ceil(self.popsize/2))
        cx3 = self.DA.X[DA1]
        cy3 = self.DA.ObjV[DA1]

        ParentC = Pop(np.vstack([cx1, cx2, cx3]))
        ParentC.ObjV = np.vstack([cy1, cy2, cy3])

        CAP3 = np.random.randint(0,self.CA.ObjV.shape[0],self.popsize)
        ParentM = Pop(self.CA.X[CAP3])
        ParentM.ObjV = self.CA.ObjV[CAP3]

        return ParentC, ParentM

    def reproduction(self, X, proC, disC, proM, disM):
        X1 = X[0: math.floor(X.shape[0] / 2), :]
        X2 = X[math.floor(X.shape[0] / 2): math.floor(X.shape[0] / 2) * 2, :]
        N = X1.shape[0]
        D = X1.shape[1]

        beta = np.zeros((N, D))
        mu = np.random.random((N, D))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        beta[np.random.random((N, D)) < 0.5] = 1
        beta[np.tile(np.random.random((N, 1)) > proC, (1, D))] = 1
        Offspring = np.vstack(
            (
                (X1 + X2) / 2 + beta * (X1 - X2) / 2,
                (X1 + X2) / 2 - beta * (X1 - X2) / 2,
            )
        )
        Lower = np.tile(self.xmin, (2 * N, 1))
        Upper = np.tile(self.xmax, (2 * N, 1))
        Site = np.random.random((2 * N, D)) < proM / D
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
                        ** (disM + 1)
                )
                ** (1 / (disM + 1))
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
                        ** (disM + 1)
                )
                ** (1 / (disM + 1))
        )
        Offspring = np.clip(Offspring, self.xmin, self.xmax)
        return Offspring


    def run(self):
        self.initlization()

        while(self.FEs < self.MaxFEs):
            # print(self.FEs)
            # MatingSelection
            ParentC, ParentM = self.matingSelection()

            oX1 = self.reproduction(ParentC.X, 1,20,0,0)
            oX2 = self.reproduction(ParentM.X, 0,0,1,20)

            oX = np.vstack([oX1, oX2])
            epop = Pop(oX)
            epop.ObjV = self.func(oX)
            self.FEs += oX.shape[0]

            self.update_CA(epop)
            self.update_DA(epop)

        return self.DA.X, self.DA.ObjV


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
    # problem = ZDT1(n_var=10)  #n_obj=3
    # ta2 = TA2(problem, 100, 1e5)
    # igd = IGD(problem.pareto_front())
    # X, F = ta2.run()
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
            alg = TA2(problem, 100, 1e5)
            X, F = alg.run()
            hv = ind(F)
            igd = IGD(problem.pareto_front())
            igd_lst.append(igd(F))
            hv_lst.append(hv/np.prod(refpoint))
        
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"TA2 算法在问题{fname}上10次独立实验目标值HV结果: mean(std)={np.mean(hv_lst):.2e}({np.std(hv_lst):.2e})")
        print(f"TA2 算法在问题{fname}上10次独立实验目标值IGD结果: mean(std)={np.mean(igd_lst):.2e}({np.std(igd_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")

