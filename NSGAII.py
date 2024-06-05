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

class NSGA2(object):
    def __init__(self, prob, popsize, MaxFEs):
        self.MaxFEs = MaxFEs
        self.popsize = popsize

        self.dim = prob.n_var
        self.xmin = prob.bounds()[0]
        self.xmax = prob.bounds()[1]
        self.func = prob.evaluate
        self.m = prob.n_obj

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

        self.pop = None
        self.FEs = 0

    def initPop(self):
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

    # 判断支配关系：p支配q，返回1
    def is_dominated(self, p, q):
        if (np.any(p > q)):
            return 0
        elif (np.all(p == q)):
            return 0
        else:
            return 1

    # 快速非支配排序
    def fast_nds(self, pop):
        popsize = pop.X.shape[0]
        sp_lst = [[] for i in range(popsize)]
        F = [[]]
        np_lst = [0 for i in range(popsize)]
        rank_lst = np.zeros(popsize).astype(int)  # 保存每个个体所处支配层

        # 寻找pareto第一级个体
        for p in range(0, popsize):
            sp_lst[p] = []
            np_lst[p] = 0
            for q in range(0, popsize):
                if (p != q):
                    if (self.is_dominated(pop.ObjV[p], pop.ObjV[q])):  # pop[i] 支配 pop[j],添加至sp_lst[i]
                        if (q not in sp_lst):
                            sp_lst[p].append(q)
                    elif (self.is_dominated(pop.ObjV[q], pop.ObjV[p])):
                        np_lst[p] = np_lst[p] + 1

            if (np_lst[p] == 0):
                rank_lst[p] = 1  # np为0，个体为pareto第一级
                if p not in F[0]:
                    F[0].append(p)
        i = 1
        while (F[i - 1] != []):
            Q = []
            for p in F[i - 1]:
                for j in sp_lst[p]:
                    np_lst[j] -= 1
                    if (np_lst[j] == 0):
                        rank_lst[j] = i + 1
                        if j not in Q:
                            Q.append(j)
            i = i + 1
            F.append(Q)
        return F, rank_lst


    def eff_nds(self, pop):

        #Use efficient non-dominated sort with sequential search (ENS-SS)
        def ENS_SS(pop_obj):
            # Use efficient non-dominated sort with sequential search (ENS-SS)
            nsort = pop_obj.shape[0]
            # 目标值重复的个体不参与非支配排序
            objv, index, ind = np.unique(
                pop_obj, return_index=True, return_inverse=True, axis=0
            )
            count, M = objv.shape
            frontno = np.full(count, np.inf)
            maxfront = 0
            # 对全部个体进行非支配排序
            Table, _ = np.histogram(ind, bins=np.arange(0, np.max(ind) + 2))
            while np.sum(Table[frontno < np.inf]) < np.min((nsort, len(ind))):
                maxfront += 1
                for i in range(count):
                    if frontno[i] == np.inf:
                        dominate = False
                        for j in range(i - 1, -1, -1):
                            if frontno[j] == maxfront:
                                m = 1
                                while m < M and objv[i][m] >= objv[j][m]:
                                    m += 1
                                dominate = m == M
                                if dominate or M == 2:
                                    break
                        if not dominate:
                            frontno[i] = maxfront
            frontno = frontno[ind]
            return [frontno, maxfront]

        #  Use tree-based efficient non-dominated sort (T-ENS)
        def  T_ENS(objv):
            pass

        N, M = pop.ObjV.shape
        if M < 3 or N < 500 :
            FrontNo, MaxFNo = ENS_SS(pop.ObjV)  #ENS-SS
        else:
            FrontNo, MaxFNo = T_ENS(pop.ObjV)  #T-ENS

        return MaxFNo, FrontNo

    def crowded_distance(self, Fi, pop):
        l = len(Fi)
        m = pop.ObjV.shape[1]

        Fi = np.array(Fi).astype(int)
        distI = np.zeros(l)  # 对应Fi上的位置
        for i in range(m):
            sortFinds = np.argsort(pop.ObjV[Fi, i])  # l个个体在fm上的排序
            distI[sortFinds[0]] = np.inf
            distI[sortFinds[-1]] = np.inf
            for j in range(1, l - 1):
                distI[sortFinds[j]] += (pop.ObjV[Fi[sortFinds[j + 1]], i] - pop.ObjV[Fi[sortFinds[j - 1]], i]) / (
                        np.max(pop.ObjV[:, i]) - np.min(pop.ObjV[:, i]))
        return distI

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

    def rm_dup(self, cpop):
        uy, uindx = np.unique(cpop.ObjV, axis=0, return_index=True)
        uX = cpop.X[uindx]
        cpop.X = uX
        cpop.ObjV = uy


    def environment_selection(self, pop):
        # 非支配排序
        _, frontno =  self.eff_nds(pop) # self.fast_nds(pop) #

        # 计算拥挤度距离
        Next = []
        nextfrontNO = np.zeros(self.popsize).astype(int)
        nextcd = np.zeros(self.popsize)

        i, c = 1, 0
        Fi = np.where(frontno == 1)[0]
        while (c + len(Fi) < self.popsize):
            Next.extend(Fi.tolist())
            nextfrontNO[c:c + len(Fi)] = i
            nextcd[c:c + len(Fi)] = self.crowded_distance(Fi, pop)
            c += len(Fi)
            i += 1
            Fi = np.where(frontno == i)[0]

        # 最后一层根据cd选
        lastcd = self.crowded_distance(Fi, pop)
        slinds = lastcd.argsort()[::-1]
        Next.extend(Fi[slinds[:self.popsize - c]].tolist())
        nextfrontNO[c:] = i
        nextcd[c:] = lastcd[slinds[:self.popsize - c]]


        return Next, nextfrontNO, nextcd


    def run(self):
        self.initPop()

        Next, frontno, cd = self.environment_selection(self.pop)
        # 更新
        self.pop.X = self.pop.X[Next]
        self.pop.ObjV = self.pop.ObjV[Next]

        while(self.FEs < self.MaxFEs):
            # print(self.FEs)
            # 锦标赛选择
            bts_inds = self.tournament_selection(2, self.popsize, frontno, -cd)
            self.pop.X = self.pop.X[bts_inds]
            self.pop.ObjV = self.pop.ObjV[bts_inds]

            # 生成子代
            newX = self.reproduction(self.pop.X)
            newf = self.func(newX)
            self.FEs += newf.shape[0]
            allX = np.vstack([self.pop.X, newX])
            cpop = Pop(allX)
            cpop.ObjV = np.vstack([self.pop.ObjV, newf])

            # 去除重复样本
            # self.rm_dup(cpop)

            Next, frontno, cd = self.environment_selection(cpop)

            self.pop.X = cpop.X[Next]
            self.pop.ObjV = cpop.ObjV[Next]

        _, frontno = self.eff_nds(self.pop)
        indxs = np.where(frontno == 1)[0]
        bX = self.pop.X[indxs]
        bf = self.pop.ObjV[indxs]

        return bX, bf

def plot_NDS(PF, F):
    fig = plt.figure()
    if PF.shape[1] == 2:
        plt.scatter(PF[:, 0], PF[:, 1], label = "Pareto Front")
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
    # nsga = NSGA2(problem, 100, 1e5)
    # igd = IGD(problem.pareto_front())
    # X, F = nsga.run()
    # print(igd(F))
    # plot_NDS(problem.pareto_front(), F)

    p_dct = {
        # 'WFG1': WFG1,   
        # 'WFG4': WFG4
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
            nsga = NSGA2(problem, 100, 1e5)
            X, F = nsga.run()
            hv = ind(F)
            igd = IGD(problem.pareto_front())
            igd_lst.append(igd(F))
            hv_lst.append(hv/np.prod(refpoint))
        
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"NSGA-II 算法在问题{fname}上10次独立实验目标值HV结果: mean(std)={np.mean(hv_lst):.2e}({np.std(hv_lst):.2e})")
        print(f"NSGA-II 算法在问题{fname}上10次独立实验目标值IGD结果: mean(std)={np.mean(igd_lst):.2e}({np.std(igd_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")


