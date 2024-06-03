import random
import numpy as np
import time
from copy import deepcopy
from problems.scop import CEC2017
from tqdm import tqdm

def WeightGenerator(popsize, conV, objF, rate, CorIndex, diversityDerta, stage):
    pmin = 1/(1 + np.exp(25 * (rate - CorIndex)))
    pmax = 1/(1 + np.exp(25 * (rate - CorIndex - diversityDerta)))

    if pmin >= pmax:
        pmax = 1e-6
        pmin = 0

    weights = np.linspace(pmin, pmax, popsize)
    normalvoi = (conV - np.min(conV)) / (np.max(conV) - np.min(conV) + 1e-15)
    normalfit = (objF - np.min(objF)) / (np.max(objF) - np.min(objF) + 1e-15)

    sort_idxs = np.argsort(normalfit / (normalvoi + 2e-16))
    weights[sort_idxs] = weights
    
    if stage == 1:
        weights = np.ones(popsize)
    
    return weights


def nahvyb_expt(N, k, expt=None):
    opora = list(range(N))  # Create a list from 0 to N-1
    if expt is not None:  # Check if expt is provided and remove it
        try:
            if isinstance(expt, list):
                for ex in expt:
                    opora.remove(ex)  # Remove the specified exempted value
            else:
                opora.remove(expt)
        except ValueError:
            pass  # If expt is not in the list, do nothing
    vyb = [0] * k  # Initialize a list of zeros with length k
    for i in range(k):
        index = random.randint(0, len(opora) - 1)  # Generate a random index
        vyb[i] = opora[index]  # Take the element at that index
        opora.pop(index)  # Remove the selected element from the list
    return vyb

def RouletteSelection(Roulette, Num=1):
    sumR = np.sum(Roulette)

    if sumR != 1:
        Roulette /= sumR
    
    index = np.zeros(Num, dtype=int)
    for i in range(Num):
        r = random.random()
        for x in range(len(Roulette)):
            if r <= sum(Roulette[:x]):
                index[i] = x
                break
    if Num == 1:
        return index[0]
    else:
        return index


def sort_EC(obj, cv, ep):
    cv[cv <= ep] = 0
    rank = np.lexsort((obj, cv))
    return rank

def sort_FP(obj, cv):
    rank = np.lexsort((obj, cv))
    return rank

def sort_SR(obj, cv, ep):
    pass


def zrcad(y, a, b):
    zrc = np.where((y < a) | (y > b))[0]
    for i in zrc:
        if y[i] > b[i]:
            y[i] = 2 * b[i] - y[i]
        elif y[i] < a[i]:
            y[i] = 2 * a[i] - y[i]
    return y

def InterCompare(conleadpop_obj, conleadpop_con, objleadpop_obj, objleadpop_con):
    mix_obj = np.append(objleadpop_obj, conleadpop_obj)
    mix_con = np.append(objleadpop_con, conleadpop_con)

    objIndex = np.argsort(mix_obj)
    conIndex = np.argsort(mix_con)
    
    popsize = len(conleadpop_obj)
    temp1 = np.where(objIndex[:popsize] > popsize)[0]
    temp2 = np.where(conIndex[:popsize] > popsize)[0]

    size1 = len(temp1)
    size2 = len(temp2)

    return size1, size2

class CCEF:
    def __init__(self, prob, popsize, maxfes):
        self.dim = prob.dim
        self.popsize = popsize
        self.prob = prob
        self.maxfes = maxfes

        self.PR0 = 0.1
        self.alpha = 0.5
        self.beta = 0.1
        self.Pmin = 0.1
        self.RH = 10

        # CHT parameters
        self.EC_Top = 0.2
        self.EC_Cp = 5
        self.EC_P = 0.8

        self.DE_beta = 6
        self.DE_alpha = 0.75
        self.DE_gama = 30
        self.DE_P = 0.85
        self.CO_LP = 0.05

        self.CHNum = 4
        self.FP_ch = 0
        self.EC_ch = 1
        self.DE_ch = 2
        self.CO_ch = 3

        # DE parameters
        self.P = 0.2
        self.STNum = 2

        self.FEs = 0

        # population
        self.pops = []  # CHNum 个种群

        self.archs = [] # CHNum 个档案
        

        self.best_so_far_x, self.best_so_far_y = None, float('inf')

    
    def initialize(self):
        x = np.random.uniform(self.prob.lb, self.prob.ub, size=(self.popsize, self.dim))  # population
        cv = np.empty(self.popsize)
        obj = np.empty(self.popsize)

        for i in range(self.popsize):
            objv, cv_total, _ = self._evaluate_fitness(x[i])
            obj[i] = objv
            cv[i] = cv_total  # 总约束违反
        
        pop = [x, obj, cv]

        return pop

    
    def Generation(self, population, unionx, STRecord, ch_k, Ep, DE_wei, CO_wei):
        F_pool = [0.6, 0.8, 1.0]
        CR_pool = [0.1, 0.2, 1.0]

        # N = len(population[0])
        rank_fp = np.lexsort((population[1], population[2]))
        
        obj,cv = population[1],population[2]
        normal_obj = (obj-np.min(obj))/(np.std(obj) + 2e-16)
        normal_cv = (cv - np.min(cv))/(np.std(cv) + 2e-16)

        if ch_k == self.FP_ch or ch_k == self.EC_ch:
            if ch_k == self.FP_ch:
                Ep = 0

            rank = sort_EC(population[1], population[2], Ep)
            pop_pbest = rank[:round(self.P * len(population[0]))]

        roulette = STRecord / np.sum(STRecord)

        offx = np.empty_like(population[0])
        offST = np.zeros(len(population[0]), dtype=int)
        
        for i in range(len(population[0])):
            offx[i] = population[0][i].copy()
            F = F_pool[np.random.randint(len(F_pool))]
            CR = CR_pool[np.random.randint(len(CR_pool))]

            # Heuristic
            roulette_temp = roulette.copy()
            if rank_fp[i] < self.popsize * 0.5:
                roulette_temp[0] = roulette_temp[0] * 0.9
                roulette_temp[1] = roulette_temp[1] * 0.1
            else:
                roulette_temp[0] = roulette_temp[0] * 0.1
                roulette_temp[1] = roulette_temp[1] * 0.9
            roulette_temp /= np.sum(roulette_temp)

            offST[i] = RouletteSelection(roulette_temp)
        
            if offST[i] == 0:
                # current-to-rand
                A = nahvyb_expt(self.popsize, 3, i)
                r1, r2, r3 = A[0], A[1], A[2]
                offx[i] = population[0][i] + random.random() * (population[0][r1] - population[0][i]) + F * (population[0][r2] - population[0][r3])
                 
            elif offST[i] == 1:
                # rand-to-pbest
                if ch_k == self.DE_ch:
                    fit = DE_wei[i] * normal_obj + (1 - DE_wei[i]) * normal_cv
                    rank = np.argsort(fit)
                    pop_pbest = rank[:round(self.P * len(population[0]))]
                elif ch_k == self.CO_ch:
                    fit = CO_wei[i] * normal_obj + (1 - CO_wei[i]) * normal_cv
                    rank = np.argsort(fit)
                    pop_pbest = rank[:round(self.P * len(population[0]))]

                pbest = pop_pbest[np.random.randint(len(pop_pbest))]
                A = nahvyb_expt(self.popsize, 2, [i, pbest])
                r1, r2 = A[0], A[1]

                B = nahvyb_expt(len(unionx), 1, [i, r1, r2, pbest])
                r3 = B[0]
                v = population[0][r1] + F * (population[0][pbest] - population[0][r1]) + F * (population[0][r2] - unionx[r3])
                change = np.where(np.random.rand(self.dim) < CR)[0]  #
                offx[i][change] = v[change]

            # 区间
            offx[i] = zrcad(offx[i], self.prob.lb, self.prob.ub)

        return offx, offST
    
                
    def _evaluate_fitness(self, x):
        obj, cv = self.prob.evaluate(x)
        cv_total = np.sum(cv)  # 总约束违反
        self.FEs += 1
        reward = False
        # update best-so-far solution (x) and fitness (y)
        if cv_total == 0 and obj < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = x, obj
            reward = True

        return obj, cv_total, reward

    def Selection_Tournament(self, pop, offx, offobj, offcv, Ep=0):
        replace_cv = (pop[2] > offcv) & (pop[2] > Ep) & (offcv > Ep) # 约束违反小
        equal_cv = (pop[2] <= Ep) & (offcv <= Ep)  # 都可行
        replace_f = pop[1] > offobj
        replace = (equal_cv & replace_f) | replace_cv

        pop[0][replace] = offx[replace]
        pop[1][replace] = offobj[replace]
        pop[2][replace] = offcv[replace]

        return replace

    def run(self):
        pop = self.initialize()
        for ch in range(self.CHNum):
            self.pops.append(deepcopy(pop))
            self.archs.append([])
        
        n0 = 2
        delta = 1 / (5 * self.STNum)
        STRecord = np.zeros(self.STNum) + n0

        # Initialize parameters of EC
        n = int(np.ceil(self.EC_Top * len(self.pops[self.EC_ch][0])))
        cv_temp = self.pops[self.EC_ch][2]
        idx = np.argsort(cv_temp)
        Ep0 = cv_temp[idx[n]]
        EC_rate = 0

        # Initialize parameters of DeMO
        VAR0 = min(10 ** self.dim/2, np.max(self.pops[self.DE_ch][2]))
        DE_Cp = (-np.log(VAR0) - self.DE_beta)/np.log(1 - self.DE_P)
        pmax = 1
        DE_rate = 0

        # Initialize parameters of COR
        CO_Flag = False
        CO_arch = deepcopy(self.pops[self.CO_ch])
        CO_rate = 0
        CO_Idx = 0
        Div_Delta = 0
        p = self.pops[self.CO_ch][0].T
        Div_Init = np.sum(np.std(p))/p.shape[1]
        Div_Idx = Div_Init.copy()
        bR1 = []
        bR2 = []

        # Initialize parameters of CCEF
        HR = np.zeros((self.CHNum, self.RH))
        HRIdx = np.zeros(self.CHNum, dtype=int)
        CHPro = 1 / self.CHNum * np.ones(self.CHNum)


        while self.FEs < self.maxfes:

            if self.FEs <= self.beta * self.maxfes:
                CHPro = 1 / self.CHNum * np.ones(self.CHNum)
            else:
                if np.sum(HR) != 0:
                    CHPro = self.Pmin / self.CHNum + (1 - self.Pmin) * np.sum(HR, axis=1)/np.max(np.sum(HR))
                else:
                    CHPro = 1 / self.CHNum * np.ones(self.CHNum)
            
            ch_k = RouletteSelection(CHPro)

            # Update parameters of EC
            if self.FEs < self.EC_P * self.maxfes:
                Ep = Ep0 * (1 - EC_rate)**self.EC_Cp
            else:
                Ep = 0
            EC_rate += self.popsize / self.maxfes

            # Update parameters of DeMO
            if DE_rate < self.DE_P:
                VAR = VAR0 * (1 - DE_rate)**DE_Cp
            else:
                VAR = 0
            DE_rate += self.popsize/self.maxfes
            if len(np.where(self.pops[self.DE_ch][2]==0)[0]) > 0.85 * len(self.pops[self.DE_ch][0]):
                VAR = 0

            rand_idx = np.random.permutation(list(range(len(self.pops[self.DE_ch][0]))))
            self.pops[self.DE_ch][0] = self.pops[self.DE_ch][0][rand_idx]
            self.pops[self.DE_ch][1] = self.pops[self.DE_ch][1][rand_idx]
            self.pops[self.DE_ch][2] = self.pops[self.DE_ch][2][rand_idx]

            if len(np.where(self.pops[self.DE_ch][2] < VAR)[0]) == 0:
                pmax = 1e-18

            pr = max(1e-18, pmax / (1 + np.exp(self.DE_gama * (self.FEs / self.maxfes - self.alpha))))
            DE_weights = np.linspace(0, pr, self.popsize)
            rand_idx = np.random.permutation(list(range(len(DE_weights))))
            DE_weights = DE_weights[rand_idx]

            # Update parameters of COR
            if self.FEs < self.CO_LP * self.maxfes:
                CO_stage = 1
            else:
                CO_stage = 2
                CO_rate += self.popsize/self.maxfes
                if not CO_Flag:
                    recordlen = len(bR1)
                    brlen1 = np.sum(np.array(bR1) != 0)
                    brlen2 = np.sum(np.array(bR2) != 0)
                    brlen = min(brlen1, brlen2)

                    CO_Idx = brlen/recordlen
                    Div_Delta = Div_Init - Div_Idx
                    CO_Flag = True
            
            CO_weights = WeightGenerator(len(self.pops[self.CO_ch][0]), self.pops[self.CO_ch][2], self.pops[self.CO_ch][1], CO_rate, CO_Idx, Div_Delta, CO_stage)
            
            # Parent Reconstruction
            parent = deepcopy(self.pops[ch_k])
            HelpCH = np.arange(self.CHNum)
            HelpCH = HelpCH[HelpCH != ch_k]

            for i in range(len(parent[0])):
                if random.random() < self.PR0:
                    ch_help = HelpCH[np.random.randint(len(HelpCH))]
                    if (parent[2][i] > self.pops[ch_help][2][i]) or ((parent[2][i] == self.pops[ch_help][2][i]) and parent[1][i] > self.pops[ch_help][1][i]):
                        parent[0][i] = self.pops[ch_help][0][i]
                        parent[1][i] = self.pops[ch_help][1][i]
                        parent[2][i] = self.pops[ch_help][2][i]

            if np.min(STRecord) / np.sum(STRecord) < delta:
                STRecord = np.zeros(self.STNum) + n0

            # Offspring Generation
            unionx = np.vstack([parent[0], self.archs[ch_k][0]]) if len(self.archs[ch_k]) > 0 else parent[0]
            offx, offST = self.Generation(parent, unionx, STRecord, ch_k, Ep, DE_weights, CO_weights)
            offobj = np.empty(offx.shape[0])
            offcv = np.empty(offx.shape[0])
            reward_global = False
            for i in range(offx.shape[0]):
                obj, cv_total, reward = self._evaluate_fitness(offx[i])
                offobj[i] = obj
                offcv[i] = cv_total
                reward_global = reward_global or reward
            
            # Calculate pop reward
            reward_pop = self.Selection_Tournament(self.pops[ch_k], offx, offobj, offcv)
            
            # Calculate strategy reward
            replace_fp = self.Selection_Tournament(parent, offx, offobj, offcv)
    
            is_used, _ = np.histogram(offST[replace_fp], bins=self.STNum)
            STRecord += is_used 
            
            # Offspring Diffusion
            for ch in range(self.CHNum):
                # Selection
                obj = np.append(self.pops[ch][1], offobj)
                cv = np.append(self.pops[ch][2], offcv)
                normal_obj = (obj-np.min(obj))/(np.std(obj) + 2e-16)
                normal_cv = (cv - np.min(cv))/(np.std(cv) + 2e-16)

                replace = [False for _ in range(len(self.pops[ch][0]))]

                size = len(obj)
                for i in range(len(self.pops[ch][0])):
                    if ch == self.FP_ch:
                        obj_pair = np.array([obj[i], obj[i + size//2 ]])
                        cv_pair = np.array([cv[i], cv[i + size//2]])
                        flag = sort_FP(obj_pair, cv_pair)
                    elif ch == self.EC_ch:
                        obj_pair = np.array([obj[i], obj[i + size//2 ]])
                        cv_pair = np.array([cv[i], cv[i + size//2]])
                        flag = sort_EC(obj_pair, cv_pair, Ep)
                    elif ch == self.DE_ch:
                        obj_pair = np.array([normal_obj[i], normal_obj[i + size//2 ]])
                        cv_pair = np.array([normal_cv[i], normal_cv[i + size//2]])
                        fit = DE_weights[i] * obj_pair + (1-DE_weights[i]) * cv_pair
                        flag = np.argsort(fit)
                    elif ch == self.CO_ch:
                        obj_pair = np.array([normal_obj[i], normal_obj[i + size//2 ]])
                        cv_pair = np.array([normal_cv[i], normal_cv[i + size//2]])
                        fit = CO_weights[i] * obj_pair + (1-CO_weights[i]) * cv_pair
                        flag = np.argsort(fit)
                    
                    replace[i] = (flag[0] != 0)
                

                # Update operator DE archive
                if self.archs[ch]:
                    self.archs[ch][0] =  np.vstack([self.archs[ch][0], self.pops[ch][0][replace]])
                    self.archs[ch][1] = np.append(self.archs[ch][1], self.pops[ch][1][replace])
                    self.archs[ch][2] = np.append(self.archs[ch][2], self.pops[ch][2][replace])
                else:
                    self.archs[ch] = [self.pops[ch][0][replace], self.pops[ch][1][replace], self.pops[ch][2][replace]]

                if len(self.archs[ch][0]) > self.popsize:
                    idx = np.random.choice(len(self.archs[ch][0]), self.popsize, replace=False)
                    self.archs[ch][0] = self.archs[ch][0][idx]
                    self.archs[ch][1] = self.archs[ch][1][idx]
                    self.archs[ch][2] = self.archs[ch][2][idx]
                
                self.pops[ch][0][replace] = offx[replace]
                self.pops[ch][1][replace] = offobj[replace]
                self.pops[ch][2][replace] = offcv[replace]
                
                # Calculate parameters of COR
                if ch == self.CO_ch and self.FEs < self.CO_LP * self.maxfes:
                    
                    # archive select
                    self.Selection_Tournament(CO_arch, offx, offobj, offcv)

                    con_obj_betterNum, obj_con_betterNum = InterCompare(CO_arch[1], CO_arch[2], self.pops[self.CO_ch][1], self.pops[self.CO_ch][2])
                    p = self.pops[self.CO_ch][0].T
                    Div_Idx = np.sum(np.std(p)) / p.shape[1]
                    bR1.append(con_obj_betterNum)
                    bR2.append(obj_con_betterNum)
            
            Rb = reward_global
            Rp = reward_pop
            HR[ch_k, HRIdx[ch_k]] = self.alpha * Rb + (1 - self.alpha) * np.mean(Rp)
            HRIdx[ch_k] = (HRIdx[ch_k] % self.RH) 

            # if self.FEs % 10 == 0:
            #     info = '  * Generation {:d}: best_so_far_y {:7.5e} & Evaluations {:d}'
            #     print(info.format(self.FEs//self.popsize, self.best_so_far_y, self.FEs))

        return self.best_so_far_x, self.best_so_far_y


if __name__ == "__main__":
    # prob = CEC2017("C05", 10)

    # ccef = CCEF(prob, 100, 1e5)

    # bestx, besty = ccef.run()
    
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
            de = CCEF(problem, 100, 1e5)
            bestx, besty = de.run()
            if bestx is not None:
                err_lst.append(besty)
        end = time.time()
        t = (end - start)/M
        ex_time += t
        print(f"CCEF 算法在问题{fname}上10次独立实验目标值结果: 可行解比例{len(err_lst)/10:.2%} mean(std)={np.mean(err_lst):.2e}({np.std(err_lst):.2e})")
        print(f"运行时间： {t:.2f} s")
    
    print("实验结束！")
    print(f"平均运行时间： {ex_time:.2f} s")







                    



            






        


