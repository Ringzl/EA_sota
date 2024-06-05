import math
import numpy as np
from scipy.special import comb
from scipy.spatial.distance import cdist
from itertools import combinations

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


class RVEA(object):
    def __init__(self, prob, popsize, MaxFEs):
        self.MaxFEs = MaxFEs
        self.popsize = popsize

        self.dim = prob.n_var
        self.m = prob.n_obj
        self.xmin = prob.bounds()[0]
        self.xmax = prob.bounds()[1]
        self.func = prob.evaluate


        self.alpha = 2
        self.fr = 0.1

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

        self.RV = None
        self.RV0 = None
        self.pop = None
        self.FEs = 0

    def NBI(self, N: int, M: int):
        """
        生成N个M维的均匀分布的权重向量
        :param N: 种群大小
        :param M: 目标维数
        :return: 返回权重向量和种群大小，种群大小可能有改变
        """
        H1 = 1
        while comb(H1 + M, M - 1, exact=True) <= N:
            H1 += 1
        W = (
                np.array(list(combinations(range(1, H1 + M), M - 1)))
                - np.tile(np.arange(M - 1), (comb(H1 + M - 1, M - 1, exact=True), 1))
                - 1
        )
        W = (
                    np.hstack((W, np.zeros((W.shape[0], 1)) + H1))
                    - np.hstack((np.zeros((W.shape[0], 1)), W))
            ) / H1
        if H1 < M:
            H2 = 0
            while (
                    comb(H1 + M - 1, M - 1, exact=True)
                    + comb(H2 + M, M - 1, exact=True)
                    <= N
            ):
                H2 += 1
            if H2 > 0:
                W2 = (
                        np.array(list(combinations(range(1, H2 + M), M - 1)))
                        - np.tile(
                    np.arange(M - 1), (comb(H2 + M - 1, M - 1, exact=True), 1)
                )
                        - 1
                )
                W2 = (
                             np.hstack((W, np.zeros((W2.shape[0], 1))))
                             - np.hstack((np.zeros((W2.shape[0], 1)), W2))
                     ) / H2
                W = np.vstack((W, W2 / 2 + 1 / (2 * M)))
        W = np.maximum(W, 1e-6)
        return W


    def initializtion(self):
        # 初始化的参考向量
        wvs = self.NBI(self.popsize, self.m)
        # 单位化
        self.RV = wvs    #/np.linalg.norm(wvs, axis = 1).reshape(-1,1)
        self.RV0 = self.RV.copy()

        # 初始化种群
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

        return Offspring

    # Reference Vector-Guided Selection Strategy
    def EnvironmentalSelection(self, cpop, theta):
        # Translate the population(目标值)
        zmin = np.min(cpop.ObjV, axis = 0)
        F_shift = cpop.ObjV - zmin

        # Calculate the smallest angle value between each vector and others
        rv_cos = 1 - cdist(self.RV, self.RV, "cosine")
        rv_cos[np.eye(rv_cos.shape[0], dtype=bool)] = 0 # 避免选中本身
        gamma = np.min(np.arccos(rv_cos), axis=1)

        # Associate each solution to a reference vector
        Angle = np.arccos(1-cdist(F_shift, self.RV, "cosine"))
        associate = np.argmin(Angle, axis=1)

        #  Select one solution for each reference vector
        sindxs = []
        for i in np.unique(associate):
            current = np.where(associate==i)[0]  # 与向量i联系的个体
            APD = (1 + self.m * theta * Angle[current, i]/gamma[i])*np.sqrt(np.sum(F_shift[current]**2, axis = 1))
            bestind = np.argmin(APD)
            sindxs.append(current[bestind])

        return sindxs

    def RV_Adaption(self):
        zmin = np.min(self.pop.ObjV, axis=0)
        zmax = np.max(self.pop.ObjV, axis=0)
        RV = self.RV0 * (zmax - zmin)
        self.RV = RV    #/np.linalg.norm(RV, axis=1).reshape(-1,1)

    def run(self):

        self.initializtion()

        while (self.FEs < self.MaxFEs):

            # 子代生成
            matingpool = np.random.randint(0, self.pop.X.shape[0], self.popsize)
            newX = self.reproduction(self.pop.X[matingpool])
            newf = self.func(newX)
            self.FEs += newf.shape[0]

            # 合并父代与子代
            allX = np.vstack([self.pop.X, newX])
            allf = np.vstack([self.pop.ObjV, newf])
            cpop = Pop(allX)
            cpop.ObjV = allf

            # 基于APD的环境选择更新当前种群
            theta = (self.FEs/self.MaxFEs) ** self.alpha
            sindxs = self.EnvironmentalSelection(cpop, theta)
            self.pop.X = cpop.X[sindxs]
            self.pop.ObjV = cpop.ObjV[sindxs]

            if math.ceil(self.FEs/self.popsize) % math.ceil(self.fr*self.MaxFEs/self.popsize) == 0:
                self.RV_Adaption()

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
    # rvea = RVEA(problem, 100, 1e5)
    # igd = IGD(problem.pareto_front())
    # X, F = rvea.run()
    # print(igd(F))
    # plot_NDS(problem.pareto_front(), F)

    p_dct = {
        # 'WFG1': WFG1,   
        # 'WFG4': WFG4
        'DTLZ2': DTLZ2,
        'DTLZ3': DTLZ3,
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
            alg = RVEA(problem, 100, 1e5)
            X, F = alg.run()
            hv = ind(F)
            igd = IGD(problem.pareto_front())
            igd_lst.append(igd(F))
            hv_lst.append(hv/np.prod(refpoint))
        
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"RVEA 算法在问题{fname}上10次独立实验目标值HV结果: mean(std)={np.mean(hv_lst):.2e}({np.std(hv_lst):.2e})")
        print(f"RVEA 算法在问题{fname}上10次独立实验目标值IGD结果: mean(std)={np.mean(igd_lst):.2e}({np.std(igd_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")
