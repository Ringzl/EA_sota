import math
import random
import numpy as np
from scipy.stats import cauchy

from problems.bounds_cop import Rastrigin

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

def cauchy_rnd(x0, gamma):
    # Generuje losowe liczby z rozkładu Cauchy’ego o parametrach x0 i gamma
    y = x0 + gamma * np.tan(np.pi * (np.random.rand() - 1/2))
    return y

def zrcad(y, a, b):
    zrc = np.where((y < a) | (y > b))[0]
    for i in zrc:
        if y[i] > b[i]:
            y[i] = 2 * b[i] - y[i]
        elif y[i] < a[i]:
            y[i] = 2 * a[i] - y[i]
    return y
        

class EA4eig:
    def __init__(self, problem, N_init, maxfes):

        self.h = 4
        
        self.N_init = N_init
        self.N = N_init
        self.Nmin = 10
        self.n0 = 2
        self.cni = np.zeros(self.h)
        self.ni = np.zeros(self.h) + self.n0
        self.nrst = 0
        self.success = np.zeros(self.h)
        
        self.succ = 0
        self.delta = 1.0 / (5.0*self.h)

        self.problem = problem
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub

        self.current_fes = 0
        self.maxfes = maxfes
        
        # IDE 参数
        self.gmax = round(self.maxfes / self.N)
        self.SRg = np.zeros((self.gmax,1)) + 2
        self.T = self.gmax/10
        self.GT =  math.floor(self.gmax / 2)
        self.gt = self.GT
        self.g = 0
        self.Tcurr = 0

        # cobide参数
        self.CBps = 0.5
        self.peig = 0.4
        self.suceig = 0
        self.ceig = 0
        self.CBF = np.zeros(self.N)
        self.CBCR = np.zeros(self.N)
        for i in range(self.N):
            self.set_CBF_CBCR(i) 
        
        # cmaes 参数
        self.sigma = (self.ub[0]-self.lb[0])/2
        self.oldPop = None
        self.eps = 1e-15
        self.mu = self.N//2
        self.weights = np.log(self.mu + 1 / 2) - np.log(np.arange(1, self.mu + 1))[np.newaxis,:].T
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.cc = (4 + self.mueff/self.dim) /(self.dim + 4 + 2*self.mueff/self.dim)
        self.cs = (self.mueff + 2)/(self.dim + self.mueff + 5)

        self.c1 = 2/((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1-self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc, self.ps = np.zeros((self.dim,1)), np.zeros((self.dim,1))
        self.B = np.eye(self.dim, self.dim) # B defines the coordinate system
        self.D = np.ones((self.dim, 1)) # Diagonal D defines the scaling
        self.CC = np.dot(self.B, np.diag(self.D ** 2) * self.B.T) # self.B @ np.diag(self.D ** 2) @ self.B.T  # Covariance matrix C
        self.invsqrtC = np.dot(self.B, np.diag(self.D ** -1) * self.B.T) # self.B @ np.diag(self.D ** -1) @ self.B.T
        self.eigeneval = 0 # Track update of B and D
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2)) # Expectation of ||N(0,I)|| == norm(randn(DIM,1))

        # jSO 参数
        self.A_size_max = round(self.N*2.6)
        self.H = 5
        self.MF = 0.3 * np.ones(self.H)
        self.MCR = 0.8 * np.ones(self.H)
        self.MF[self.H-1] = 0.9
        self.MCR[self.H-1] = 0.9
        self.k = 0
        self.Asize = 0
        self.A = []
        self.pmax = 0.25
        self.pmin = self.pmax/2

        self.x = None
        self.y = None
        self.best_so_far_y, self.best_so_far_x = np.Inf, None


    def set_CBF_CBCR(self, i):
        if random.random() < 0.5:
            self.CBF[i] = cauchy_rnd(0.65, 0.1)
        else:
            self.CBF[i] = cauchy_rnd(1, 0.1)
        while self.CBF[i] < 0:
            if random.random() < 0.5:
                self.CBF[i] = cauchy_rnd(0.65, 0.1)
            else:
                self.CBF[i] = cauchy_rnd(1, 0.1)  
        if self.CBF[i] > 1:
            self.CBF[i] = 1
        if random.random() < 0.5:
            self.CBCR[i] = cauchy_rnd(0.1, 0.1)
        else:
            self.CBCR[i] = cauchy_rnd(0.95, 0.1)
        self.CBCR[i] = np.clip(self.CBCR[i], 0, 1)  # Ensuring CBCR is within [0, 1]

    def initialize(self):
        self.x = np.random.uniform(self.lb, self.ub, size=(self.N, self.dim))  # population
        # self.x = np.empty((self.N, self.dim))
        # for j in range(self.dim):
        #     for i in range(self.N):
        #         self.x[i, j] = self.lb[j] + np.random.uniform(i / self.N * (self.ub[j] - self.lb[j]), (i + 1) / self.N * (self.ub[j] - self.lb[j]))
        #     np.random.shuffle(self.x[:, j])
        
        self.y = np.empty((self.N,))  # fitness
        for i in range(self.N):
            self.y[i] = self._evaluate_fitness(self.x[i])
        self.oldPop = self.x.copy().T

    
    def _evaluate_fitness(self, x):
        y = self.problem.objective_function(x)
        self.current_fes += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        return float(y)


    def cobide(self, hh):

        ox = np.empty_like(self.x)
        oy = np.empty(self.N)

        if random.random() < self.peig: #
            self.ceig = 1
            sort_idxs = self.y.argsort()
            PopeigX = self.x[sort_idxs][:round(self.N * self.CBps), :]
            # Popeigy = self.y[sort_idxs][:int(self.N * self.CBps)]
            cov_matrix = np.cov(PopeigX, rowvar=False)
            _, EigVect = np.linalg.eigh(cov_matrix)
        
            for i in range(self.N):
                vyb = nahvyb_expt(self.N, 3, i)
                xr1 = self.x[vyb[0], :].copy()
                xr2 = self.x[vyb[1], :].copy()
                xr3 = self.x[vyb[2], :].copy()

                v =  xr1 + self.CBF[i] * (xr2 - xr3)
                v = zrcad(v, self.lb, self.ub)
                
                xi = self.x[i, :].copy()
                xieig = np.dot(EigVect.T, xi[np.newaxis,:].T)
                veig = np.dot(EigVect.T, v[np.newaxis,:].T)
                
                change = np.where(np.random.rand(self.dim) < self.CBCR[i])[0]
                if len(change) == 0:
                    change = np.array([math.floor(self.dim * random.random())])
                xieig[change] = veig[change]
                xi = np.dot(EigVect, xieig).T.squeeze()
                xi = zrcad(xi, self.lb, self.ub)
                yi = self._evaluate_fitness(xi)
                ox[i] = xi
                oy[i] = yi
        else:
            
            for i in range(self.N):
                vyb = nahvyb_expt(self.N, 3, i)
                xr1 = self.x[vyb[0], :].copy()
                xr2 = self.x[vyb[1], :].copy()
                xr3 = self.x[vyb[2], :].copy()
                v =  xr1 + self.CBF[i] * (xr2 - xr3)  #
                v = zrcad(v, self.lb, self.ub)

                xi = self.x[i, :].copy()
                change = np.where(np.random.rand(self.dim) < self.CBCR[i])[0]  #
                if len(change) == 0:
                    change = np.array([math.floor(self.dim * random.random())])
                xi[change] = v[change]
                xi = zrcad(xi, self.lb, self.ub)
                yi = self._evaluate_fitness(xi)
                ox[i] = xi
                oy[i] = yi
                
        if self.ceig == 1:
            for i in range(self.N):
                if oy[i] <= self.y[i]:
                    self.x[i,:] = ox[i,:].copy()
                    self.y[i] = oy[i]
                    self.suceig += 1
                    self.success[hh] += 1
                    self.ni[hh] += 1
                else:
                    self.set_CBF_CBCR(i)
            self.ceig = 0
        else:
            for i in range(self.N):
                if oy[i] <= self.y[i]:
                    self.x[i,:] = ox[i,:].copy()
                    self.y[i] = oy[i]
                    self.success[hh] += 1
                    self.ni[hh] += 1
                else:
                    self.set_CBF_CBCR(i)
    
    def ide(self, hh):
        sort_idxs = self.y.argsort()
        self.x = self.x[sort_idxs]
        self.y = self.y[sort_idxs]
        self.CBF = self.CBF[sort_idxs]
        self.CBCR = self.CBCR[sort_idxs]

        ox = self.x.copy()
        oy = self.y.copy()
        IDEps = 0.1 + 0.9 * 10 ** (5 * (self.current_fes/self.maxfes - 1))  #self.g/self.gmax
        pd = 0.1 * IDEps
        
        if self.g < self.gt:
            SRT = 0
        else:
            SRT = 0.1
  
        for i in range(self.N):
            vyb = nahvyb_expt(self.N, 4, i)
            o = vyb[0]
            if self.g <= self.gt:
                o = i
            xo = self.x[o]
            r1, r2, r3 = vyb[1], vyb[2], vyb[3]
            xr1, xr2, xr3 = self.x[r1], self.x[r2], self.x[r3].copy()
            indperturb = np.where(np.random.rand(self.dim) < pd)[0]
            pom = np.array(self.lb) + np.random.rand(self.dim) * (np.array(self.ub) - np.array(self.lb))
            xr3[indperturb] = pom[indperturb]

            Fo = o/self.N + 0.1 * np.random.randn(1)
            while Fo <= 0 or Fo > 1:
                Fo = o/self.N + 0.1 * np.random.randn(1)
            high_ind_S = math.floor(IDEps * self.N) - 1
            if o > high_ind_S:
                if r1 > high_ind_S:
                    candidates = list(set(range(high_ind_S)) - {*vyb, i})
                    r1 = math.floor(random.random() * len(candidates))
                    xr1 = self.x[r1, :]

            if self.g > self.gt and random.random() < 0.5:
                ox[i, :] = self.x[i, :] + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
            else:
                ox[i, :] = xo + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
        
        if random.random() < self.peig:
            self.ceig = 1
            sort_idxs = self.y.argsort()
            PopeigX = self.x[sort_idxs][:round(self.N * self.CBps), :]
            # Popeigy = self.y[sort_idxs][:int(self.N * self.CBps)]
            
            cov_matrix = np.cov(PopeigX, rowvar=False)
            _, EigVect = np.linalg.eigh(cov_matrix)
            
            for i in range(self.N):
                xi = self.x[i, :].copy()
                v = ox[i, :].copy()

                xieig = np.dot(EigVect.T, xi[np.newaxis,:].T)
                veig = np.dot(EigVect.T, v[np.newaxis,:].T)
                change = np.where(np.random.rand(self.dim) < self.CBCR[i])[0]
                if len(change) == 0:
                    change = np.array([math.floor(self.dim * random.random())])
                xieig[change] = veig[change]
                xi = np.dot(EigVect, xieig).T.squeeze()
                ox[i,:] = zrcad(xi, self.lb, self.ub)
        else:
            self.ceig = 0
            for i in range(self.N):
                CR = (i+1)/self.N + 0.1 * np.random.randn()
                while CR < 0 or CR > 1:
                    CR = (i+1)/self.N + 0.1 * np.random.randn()
                jrand = math.floor(random.random() * self.dim)
                for j in range(self.dim):
                    if not (random.random() <= CR or j == jrand):
                        ox[i,j] = self.x[i,j].copy()
                    if ox[i,j] < self.lb[j] or ox[i,j] > self.ub[j]:
                        ox[i,j] = self.lb[j] + random.random() * (self.ub[j] - self.lb[j])
        
        for i in range(self.N):
            oy[i] = self._evaluate_fitness(ox[i,:])

        indsucc = np.where(oy <= self.y)[0]
        self.success[hh] += len(indsucc)
        self.ni[hh] += len(indsucc)
        SR = len(indsucc)/self.N

        if self.g < self.gt:
            if SR <= SRT:
                self.Tcurr += 1
            else:
                self.Tcurr = 0
            
            if self.Tcurr >= self.T:
                self.gt = self.g

        self.x[indsucc, :] = ox[indsucc, :].copy()
        self.y[indsucc] = oy[indsucc].copy()

        sort_idxs = self.y.argsort()
        self.x = self.x[sort_idxs]
        self.y = self.y[sort_idxs]
        self.CBF = self.CBF[sort_idxs]
        self.CBCR = self.CBCR[sort_idxs]

        self.g += 1


    def cmaes(self, hh):
        sort_idxs = self.y.argsort()
        self.x = self.x[sort_idxs]
        self.y = self.y[sort_idxs]
        self.CBF = self.CBF[sort_idxs]
        self.CBCR = self.CBCR[sort_idxs]
        
        xmean = np.dot(self.x[:self.mu, :].T, self.weights)
        Pop = np.zeros((self.dim, self.N))
        PopFit = np.zeros(self.N)
        for kk in range(self.N):
            tmp = xmean + self.sigma * np.dot(self.B, self.D * np.random.randn(self.dim, 1))
            Pop[:, kk] = tmp.squeeze()
            zrca = np.where(Pop[:, kk] < self.lb)[0]
            if len(zrca) > 0:
                Pop[zrca, kk] = (self.oldPop[zrca, kk] + np.array(self.lb)[zrca])/2
                zrcb = np.where(Pop[:, kk] > self.ub)[0]
                Pop[zrcb, kk] = (self.oldPop[zrcb, kk] + np.array(self.lb)[zrcb])/2

            PopFit[kk] = self._evaluate_fitness(Pop[:, kk])
            maxind = np.argmax(self.y)
            maxf = self.y[maxind]
            if PopFit[kk] < maxf:
                self.x[maxind, :] = Pop[:, kk]
                self.y[maxind] = PopFit[kk]
                self.success[hh] += 1
                self.ni[hh] += 1

        FitInd = np.argsort(PopFit)
        xold = xmean
        xmean = np.dot(Pop[:, FitInd[:self.mu]], self.weights)
        self.oldPop = self.x.copy().T

        # Cumulation: Update evolution paths
        self.ps = (1-self.cs) * self.ps + math.sqrt(self.cs * (2-self.cs)*self.mueff)*np.dot(self.invsqrtC, (xmean - xold)/self.sigma)
        hsig = np.sum(self.ps ** 2) / (1 - (1 - self.cs) ** (2 * self.current_fes / self.N)) / self.dim < 2 + 4 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - xold) / self.sigma

        # Adapt covariance matrix C
        artmp = (1 / self.sigma) * (Pop[:, FitInd[:self.mu]] - np.tile(xold, (1, self.mu)))  # mu difference vectors
        self.CC = (1 - self.c1 - self.cmu) * self.CC + self.c1 * (np.dot(self.pc, self.pc.T) + (1 - hsig) * self.cc * (2 - self.cc) * self.CC) + self.cmu * np.dot(artmp, np.diag(self.weights) * artmp.T)
        
        # Adapt step size sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        if self.sigma > 1e+300 or self.sigma < 1e-300 or np.isnan(self.sigma):
            self.sigma = (self.ub[0] - self.lb[0]) / 2
        
        # Update B and D from C
        if self.current_fes - self.eigeneval > self.N / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.current_fes
            self.CC = np.triu(self.CC) + np.triu(self.CC, 1).T  # enforce symmetry
            self.D, self.B = np.linalg.eigh(self.CC)  # eigen decomposition, note: eigh returns eigenvalues in ascending order
            self.D = self.D[np.newaxis,:].T
            self.D = np.sqrt(np.diag(self.D))  # contains standard deviations now
            self.invsqrtC = np.dot(self.B, np.diag(self.D ** -1) * self.B.T)

    def jso(self, hh):
        Fpole = -1 * np.ones(self.N)
        CRpole = -1 * np.ones(self.N)
        SCR, SF = [], []
        suc = 0
        ox = np.zeros((self.N, self.dim))
        oy = np.zeros(self.N)

        deltafce = -1 * np.ones(self.N)
        pp = self.pmax - (self.pmax - self.pmin) * (self.current_fes / self.maxfes)

        if random.random() < self.peig:
            self.ceig = 1
            sort_idxs = self.y.argsort()
            PopeigX = self.x[sort_idxs][:round(self.N * self.CBps), :]
            # Popeigy = self.y[sort_idxs][:int(self.N * self.CBps)]
            self.x[:self.mu, :]
            cov_matrix = np.cov(PopeigX, rowvar=False)
            _, EigVect = np.linalg.eigh(cov_matrix)

            for i in range(self.N):
                rr = nahvyb_expt(self.H, 1)[0]
                CR = self.MCR[rr] + math.sqrt(0.1) * np.random.randn()
                CR = np.clip(CR, 0, 1)

                if self.current_fes < 0.25*self.maxfes:
                    CR = max(CR, 0.7)
                elif self.current_fes < 0.5*self.maxfes:
                    CR = max(CR, 0.6)
                F = -1
                while F<=0:
                    F = random.random() * math.pi - math.pi/2
                    F = 0.1 * np.tan(F) + self.MF[rr]
                if F > 1:
                    F = 1
                if self.current_fes < 0.6*self.maxfes and F > 0.7:
                    F = 0.7
                Fpole[i] = F
                CRpole[i] = CR

                expt = [i]
                p = max(2, math.ceil(pp * self.N))
                pomx = self.x.copy()
                sort_idxs = self.y.argsort()
                pomx = pomx[sort_idxs]
                pbest = pomx[:p, :]
                ktery = math.floor(p * random.random())
                xpbest = pbest[ktery, :]

                xi = self.x[expt, :]
                vyb = nahvyb_expt(self.N, 1, expt)
                r1 = self.x[vyb, :]
                expt = expt + vyb

                vyb = nahvyb_expt(self.N+self.Asize, 1, expt)
                sjed = np.vstack([self.x, np.array(self.A)]) if self.Asize > 0 else self.x.copy()
                r2 = sjed[vyb, :]

                if self.current_fes < 0.2 * self.maxfes:
                    Fw = 0.7 * F
                elif self.current_fes < 0.4 * self.maxfes:
                    Fw = 0.8 * F
                else:
                    Fw = 1.2 * F

                v = xi + Fw * (xpbest - xi) + F * (r1 - r2)

                xieig = np.dot(EigVect.T, xi.reshape(-1,1))
                veig = np.dot(EigVect.T, v.reshape(-1,1))
                change = np.where(np.random.rand(self.dim) < self.CBCR[i])[0]
                if len(change) == 0:
                    change = math.floor(self.dim*random.random())
                xieig[change] = veig[change]
                xi = np.dot(EigVect, xieig).T.squeeze()
                ox[i,:] = zrcad(xi, self.lb, self.ub)
 
        else:
            self.ceig = 0
            for i in range(self.N):
                rr = nahvyb_expt(self.H, 1)[0]
                CR = self.MCR[rr] + math.sqrt(0.1) * np.random.randn()
                CR = np.clip(CR, 0, 1)

                if self.current_fes < 0.25*self.maxfes:
                    CR = max(CR, 0.7)
                elif self.current_fes < 0.5*self.maxfes:
                    CR = max(CR, 0.6)
                F = -1
                while F<=0:
                    F = random.random() * math.pi - math.pi/2
                    F = 0.1 * np.tan(F) + self.MF[rr]
                if F > 1:
                    F = 1

                if self.current_fes < 0.6 * self.maxfes and F > 0.7:
                    F = 0.7
                
                Fpole[i] = F
                CRpole[i] = CR

                expt = [i]
                p = max(2, math.ceil(pp * self.N))
                pomx = self.x.copy()
                sort_idxs = self.y.argsort()
                pomx = pomx[sort_idxs]
                pbest = pomx[:p, :]
                ktery = math.floor(p * random.random())
                xpbest = pbest[ktery, :]

                xi = self.x[expt, :]
                vyb = nahvyb_expt(self.N, 1, expt)
                r1 = self.x[vyb, :]
                expt = expt + vyb

                vyb = nahvyb_expt(self.N+self.Asize, 1, expt)
                sjed = np.vstack([self.x, np.array(self.A)]) if self.Asize > 0 else self.x.copy()
                r2 = sjed[vyb, :]

                if self.current_fes < 0.2 * self.maxfes:
                    Fw = 0.7 * F
                elif self.current_fes < 0.4 * self.maxfes:
                    Fw = 0.8 * F
                else:
                    Fw = 1.2 * F

                v = xi + Fw * (xpbest - xi) + F * (r1 - r2)
                change = np.where(np.random.rand(self.dim) < self.CBCR[i])[0]
                if len(change) == 0:
                    change = math.floor(self.dim*random.random())
                xi[:, change] = v[:, change]
                xi = zrcad(xi.squeeze(), self.lb, self.ub)
                ox[i,:] = xi
        
        for i in range(self.N):
            oy[i] = self._evaluate_fitness(ox[i])
        
        for i in range(self.N):
            if oy[i] < self.y[i]:
                deltafce[i] = self.y[i] - oy[i]
                suc += 1
                if self.Asize < self.A_size_max:
                    self.A.append(self.x[i])
                    self.Asize += 1
                else:
                    ktery = nahvyb_expt(self.Asize, 1)[0]
                    self.A[ktery] = self.x[i]
                SCR.append(CRpole[i])
                SF.append(Fpole[i])
            
            if oy[i] <= self.y[i]:
                self.x[i] = ox[i]
                self.y[i] = oy[i]
        
        if suc > 0:
            MCR_old = self.MCR[self.k]
            MF_old = self.MF[self.k]
            platne = np.where(deltafce != -1)[0]
            delty = deltafce[platne]
            sum_delty = np.sum(delty)
            vahyw = 1/sum_delty * delty
            mSCR = np.max(SCR)
            if self.MCR[self.k] == -1 or mSCR == 0:
                self.MCR[self.k] = -1
            else:
                meanSCRpomjm = vahyw * np.array(SCR)
                meanSCRpomci = meanSCRpomjm * np.array(SCR)
                self.MCR[self.k] = np.sum(meanSCRpomci)/np.sum(meanSCRpomjm)
            meanSCRpomjm = vahyw * np.array(SF)
            meanSCRpomci = meanSCRpomjm*np.array(SF)
            self.MF[self.k] =  np.sum(meanSCRpomci)/np.sum(meanSCRpomjm)
            self.MCR[self.k] = ( MCR_old + self.MCR[self.k])/2
            self.MF[self.k] = ( MF_old + self.MF[self.k])/2
            self.k += 1
            if self.k >= self.H:
                self.k = 1
        
        self.success[hh] += suc
        self.ni[hh] += suc
        
    def roulete(self):
        ss = np.sum(self.ni)
        p_min = np.min(self.ni)/ss

        cp = np.cumsum(self.ni)
        cp /= ss

        res = np.sum(cp < np.random.rand()) 
        return res, p_min

    def optimize(self):
        self.initialize()
        
        while self.current_fes < self.maxfes:

            hh, p_min = self.roulete()
                
            if p_min < self.delta:
                self.cni = self.cni + self.ni - self.n0
                self.ni = np.zeros(self.h) + self.n0
                self.nrst = self.nrst + 1

            if hh == 0:
                """
                1 generation of cobide
                """
                self.cobide(hh)
            if hh == 1:
                """
                1 generation of IDE
                """
                self.ide(hh)
                
            elif hh == 2:
                """
                1 generation of CMAES
                """
                self.cmaes(hh)
            elif hh == 3:
                """
                1 generation of jSO generation
                """
                self.jso(hh)

            optN = round((self.Nmin - self.N_init)/self.maxfes * self.current_fes + self.N_init)

            if self.N > optN:
                diffPop = self.N - optN
                if self.N - diffPop < self.Nmin:
                    diffPop = self.N - self.Nmin
                
                self.N -= diffPop
                sort_idxs = self.y.argsort()
                self.x, self.y = self.x[sort_idxs], self.y[sort_idxs]
                self.CBCR = self.CBCR[sort_idxs]
                self.CBF = self.CBF[sort_idxs]

                self.x = self.x[:self.N, :]
                self.y = self.y[:self.N]
                self.CBCR = self.CBCR[:self.N]
                self.CBF = self.CBF[:self.N]

                self.A_size_max = round(self.N * 2.6)
                while self.Asize > self.A_size_max:
                    index_v_arch = nahvyb_expt(self.Asize, 1)[0]
                    self.A.pop(index_v_arch)
                    self.Asize -= 1
                
                # CMAES
                self.mu = math.floor(self.N/2)
                self.weights = np.log(self.mu + 1 / 2) - np.log(np.arange(1, self.mu + 1))[np.newaxis,:].T
                self.weights = self.weights / np.sum(self.weights)
                self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)  #

            print(self.best_so_far_y)
            

if __name__ == "__main__":

    problem = Rastrigin()

    alg = EA4eig(problem, 100, 1e4)
    bestx, besty = alg.optimize()

    print(f"Best x: {bestx}, Best y: {besty}")

        






            











    