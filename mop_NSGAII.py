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
from pymoo.indicators.igd_plus import IGDPlus

import warnings
warnings.filterwarnings("ignore")
from pymoo.config import Config
Config.warnings['not_compiled'] = False


# 判断支配关系：p支配q，返回1
def is_dominated(p, q):
    if (np.any(p > q)):
        return 0
    elif (np.all(p == q)):
        return 0
    else:
        return 1

# 快速非支配排序
def fast_nds(objv):
    '''
    F: 每一级个体的索引 
    rank_lst: 对应个体的废止配等级
    '''
    size = len(objv)
    sp_lst = [[] for i in range(size)]
    F = [[]]
    np_lst = [0 for i in range(size)]
    rank_lst = np.zeros(size).astype(int)  # 保存每个个体所处支配层

    # 寻找pareto第一级个体
    for p in range(0, size):
        sp_lst[p] = []
        np_lst[p] = 0
        for q in range(0, size):
            if (p != q):
                if (is_dominated(objv[p], objv[q])):  # pop[i] 支配 pop[j],添加至sp_lst[i]
                    if (q not in sp_lst):
                        sp_lst[p].append(q)
                elif (is_dominated(objv[q], objv[p])):
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
    return rank_lst, F 

# Use efficient non-dominated sort with sequential search (ENS-SS)
def ENS_SS(pop_obj, nSort):
    # 目标值重复的个体不参与非支配排序
    objv, _, Loc = np.unique(
        pop_obj, return_index=True, return_inverse=True, axis=0
    )
    Table, _ = np.histogram(Loc, bins=np.arange(0, np.max(Loc) + 2))
    N, M = objv.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo  = -1

    # 对全部个体进行非支配排序
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(Loc)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                dominated = False
                for j in range(i - 1, -1, -1):
                    if FrontNo[j] == MaxFNo:
                        m = 1
                        while m < M and objv[i, m] >= objv[j, m]:
                            m += 1
                        dominated = m >= M
                        if dominated or M == 2:
                            break
                if not dominated:
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[Loc]
    return [FrontNo, MaxFNo]

# Use tree-based efficient non-dominated sort (T-ENS)
def T_ENS(pop_obj, nSort):
    # 目标值重复的个体不参与非支配排序
    objv, _, Loc = np.unique(
        pop_obj, return_index=True, return_inverse=True, axis=0
    )
    Table, _ = np.histogram(Loc, bins=np.arange(0, np.max(Loc) + 2))
    N, M = objv.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo  = -1

    Forest = np.full(N, np.inf)
    Children = np.zeros((N, M-1))
    LeftChild = np.zeros(N) + M
    Father = np.zeros(N)
    Brother = np.zeros(N) + M

    ORank = np.argsort(-pop_obj[:, 1:M], axis=1)
    ORank += 1

    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(Loc)):
        MaxFNo += 1
        root = np.where(FrontNo == np.inf)[0][0]
        Forest[MaxFNo] = root
        FrontNo[root] = MaxFNo
        for p in range(N):
            if FrontNo[p] == np.inf:
                Pruning = np.zeros(N)
                q = Forest[MaxFNo]
            while True:
                m = 0
                while m < M and pop_obj[p, ORank[q,m]] >= pop_obj[q, ORank[q,m]]:
                    m+=1
                if m == M-1:
                    break
                else:
                    Pruning[q] = m
                    if LeftChild[q] <= Pruning[q]:
                        q = Children[q, LeftChild[q]]
                    else:
                        while Father[q] and Brother[q] > Pruning[Father[q]]:
                            q = Father[q]
                        if Father[q]:
                            q = Children[Father[q], Brother[q]]
                        else:
                            break
            if m < M-1:
                FrontNo[p] = MaxFNo
                q = Forest[MaxFNo]
                while Children[q, Pruning[q]]:
                    q = Children[q, Pruning[q]]
                Children[q, Pruning[q]] = p
                Father[p] = q

                if LeftChild[q] > Pruning[q]:
                    Brother[p] = LeftChild[q]
                    LeftChild[q] = Pruning[q]
                else:
                    bro = Children[q, LeftChild[q]]
                    while Brother[bro] < Pruning[q]:
                        bro = Children[q, Brother[bro]]
                    Brother[p] = Brother[bro]
                    Brother[bro] = Pruning[q]
    FrontNo = FrontNo[:, Loc]
    return FrontNo, MaxFNo


# 进行K-锦标赛选择， args 输入objv，或rank
def tournament_selection(K, N, *args):
    for i in range(len(args)):
        args[i].reshape(1, -1)
    fitness = np.vstack(args)
    _, rank = np.unique(fitness, return_inverse=True, axis=1)
    parents = np.random.randint(low=0, high=len(rank), size=(N, K))
    best = np.argmin(rank[parents], axis=1)
    index = parents[np.arange(N), best]
    return index

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
        
        # 实数编码的遗传算法
        proC = 1
        disC = 20
        proM = 1
        disM = 20

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

    def eff_nds(self, pop):
        N, M = pop.ObjV.shape

        if M < 3 or N < 500 :
            FrontNo, MaxFNo = ENS_SS(pop.ObjV, self.popsize)  #ENS-SS
        else:
            FrontNo, MaxFNo = T_ENS(pop.ObjV, self.popsize)  #T-ENS

        return FrontNo, MaxFNo

    def crowded_distance(self, pop_obj,FrontNo):
        N, M = pop_obj.shape
        CrowdDis = np.zeros(N)
        Fronts = np.unique(FrontNo[FrontNo != np.inf])

        for f in range(len(Fronts)):
            Front = np.where(FrontNo == Fronts[f])[0]
            Fmax = np.max(pop_obj[Front, :], axis=0)
            Fmin = np.min(pop_obj[Front, :], axis=0)
            for i in range(M):
                Rank = np.argsort(pop_obj[Front, i])
                CrowdDis[Front[Rank[0]]] = np.inf
                CrowdDis[Front[Rank[-1]]] = np.inf
                for j in range(1,len(Front)-1):
                    CrowdDis[Front[Rank[j]]] += (pop_obj[Front[Rank[j+1]], i] - pop_obj[Front[Rank[j-1]], i])/(Fmax[i] - Fmin[i])
        return CrowdDis    
    

    def environment_selection(self, pop, N):
        # 非支配排序
        # FrontNo, _ = fast_nds(pop)
        FrontNo, MaxFNo =  self.eff_nds(pop) 
        Next = FrontNo < MaxFNo

        # 计算拥挤度距离
        CrowdDis = self.crowded_distance(pop.ObjV, FrontNo)

        # 基于拥挤度距离选择最后一层的个体
        Last = np.where(FrontNo == MaxFNo)[0]
        Rank = np.argsort(-CrowdDis[Last])
        Next[Last[Rank[:N-np.sum(Next)]]] = True

        # 更新下一代种群
        self.pop.X = pop.X[Next]
        self.pop.ObjV = pop.ObjV[Next]

        FrontNo = FrontNo[Next]
        CrowdDis = CrowdDis[Next]

        return FrontNo, CrowdDis


    def run(self):
        self.initPop()
        FrontNo,CrowdDis = self.environment_selection(self.pop, self.popsize)

        while(self.FEs < self.MaxFEs):
            # 锦标赛选择
            index = tournament_selection(2, self.popsize, FrontNo, -CrowdDis)
            parentx = self.pop.X[index]
            # 生成子代
            newX = self.reproduction(parentx)
            newf = self.func(newX)
            self.FEs += len(newf)
            cpop = Pop(np.vstack([self.pop.X, newX]))
            cpop.ObjV = np.vstack([self.pop.ObjV, newf])

            FrontNo,CrowdDis = self.environment_selection(cpop, self.popsize)

        frontno, _ = self.eff_nds(self.pop)
        indxs = np.where(frontno == 0)[0]
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
    # igd = IGD(problem.pareto_front()) # IGD
    # X, F = nsga.run()
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


