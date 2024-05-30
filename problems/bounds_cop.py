import numpy as np

from opfunu.cec_based.cec2022 import *


class TestProblem:  # RC04
    def __init__(self):
        self.dim = 4
        self.ub = [100, 100, 100, 100]
        self.lb = [-100, -100, -100, -100]

    def objective_function(self, x):
        y = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2
        return y
    
class Elliposoid:
    def __init__(self, dim=10):
        self.dim = dim
        self.ub = [100 for _ in range(self.dim)]
        self.lb = [-100 for _ in range(self.dim)]

    def objective_function(self, x):
        f = np.sum([(i + 1) * x[i] ** 2 for i in range(self.dim)])
        return f
    
class Rosenbrock:
    def __init__(self, dim=10):
        self.dim = dim
        self.ub = [100 for _ in range(self.dim)]
        self.lb = [-100 for _ in range(self.dim)]

    def objective_function(self, x):
        f = 0
        for k in range(self.dim - 1):
            f += (100 * (x[k] ** 2 - x[k + 1]) ** 2 + (x[k] - 1) ** 2)
        return f

class Alckey:
    def __init__(self, dim=10):
        self.dim = dim
        self.ub = [100 for _ in range(self.dim)]
        self.lb = [-100 for _ in range(self.dim)]
    
    def objective_function(self, x):
        z = 0.32 * x 
        f = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt((1 / self.dim) * np.sum(z ** 2))) - np.exp(
                (1 / self.dim) * np.sum(np.cos(2 * np.pi * z)))
        
        return f
    
class Rastrigin:
    def __init__(self, dim=10):
        self.dim = dim
        self.ub = [100 for _ in range(self.dim)]
        self.lb = [-100 for _ in range(self.dim)]

    def objective_function(self, x):
        z = 0.05 * x
        f = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
        return f


class CEC2022:
    def __init__(self, f_name, dim):
        
        self.dim = dim

        if f_name == "F1":
            self.func = F12022(ndim=dim)
        elif f_name == "F2":
            self.func = F22022(ndim=dim)
        elif f_name == "F3":
            self.func = F32022(ndim=dim)
        elif f_name == "F4":
            self.func = F42022(ndim=dim)
        elif f_name == "F5":
            self.func = F52022(ndim=dim)
        elif f_name == "F6":
            self.func = F62022(ndim=dim)
        elif f_name == "F7":
            self.func = F72022(ndim=dim)
        elif f_name == "F8":
            self.func = F82022(ndim=dim)
        elif f_name == "F9":
            self.func = F92022(ndim=dim)
        elif f_name == "F10":
            self.func = F102022(ndim=dim)
        elif f_name == "F11":
            self.func = F112022(ndim=dim)
        elif f_name == "F12":
            self.func = F122022(ndim=dim)

        self.ub = self.func.ub
        self.lb = self.func.lb

    def objective_function(self, x):
        return self.func.evaluate(x)
    


if __name__ == "__main__":
    prob = CEC2022("F1", 10)
    print(prob.func.f_bias)
    print(prob.objective_function(prob.func.create_solution()))
