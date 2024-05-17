import numpy as np



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
