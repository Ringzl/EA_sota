import numpy as np


class CEC2017:
    def __init__(self, fname, dim):
        self.dim = dim

        if fname == "C01":
            self.evaluate = self.C01
            self.lb = np.array([-100 for _ in range(self.dim)])
            self.ub = np.array([100 for _ in range(self.dim)])
        
        elif fname == "C02":
            self.evaluate = self.C02
            self.lb = np.array([-100 for _ in range(self.dim)])
            self.ub = np.array([100 for _ in range(self.dim)])
        
        elif fname == "C03":
            self.evaluate = self.C03
            self.lb = np.array([-100 for _ in range(self.dim)])
            self.ub = np.array([100 for _ in range(self.dim)])

        elif fname == "C04":
            self.evaluate = self.C04
            self.lb = np.array([-10 for _ in range(self.dim)])
            self.ub = np.array([10 for _ in range(self.dim)])
        
        elif fname == "C05":
            self.evaluate = self.C05
            self.lb = np.array([-10 for _ in range(self.dim)])
            self.ub = np.array([10 for _ in range(self.dim)])
        
    
    def C01(self, x):
        shift_data = np.loadtxt("/home/yongcun/work/optimize/ec/problems/data/shift_data_1.txt")
        o = shift_data[:self.dim]
        z = x - o
        f = 0
        for i in range(self.dim):
            f += np.sum([z[j] for j in range(i)])**2

        g = [np.sum(z**2 - 5000*np.cos(0.1*np.pi*z) - 4000)]
        h = [0]
        con = np.array(g + h)
        con[con<0] = 0

        return f, con
    
    def C02(self, x):
        shift_data = np.loadtxt("/home/yongcun/work/optimize/ec/problems/data/shift_data_2.txt")
        o = shift_data[:self.dim]
        z = x - o
        M = np.loadtxt(f"/home/yongcun/work/optimize/ec/problems/data/M_2_D{self.dim}.txt")
        y = np.dot(M, z.reshape(-1,1)).squeeze()
        f = 0
        for i in range(self.dim):
            f += np.sum([z[j] for j in range(i)])**2

        g = [np.sum(y**2 - 5000*np.cos(0.1*np.pi*y) - 4000)]
        h = [0]
        con = np.array(g + h)
        con[con<0] = 0

        return f, con
    
    def C03(self, x):
        shift_data = np.loadtxt("/home/yongcun/work/optimize/ec/problems/data/shift_data_3.txt")
        o = shift_data[:self.dim]
        z = x - o
        f = 0
        for i in range(self.dim):
            f += np.sum([z[j] for j in range(i)])**2

        g = [np.sum(z**2 - 5000*np.cos(0.1*np.pi*z) - 4000)]
        h = [np.abs(-np.sum(z*np.sin(0.1*np.pi*z))) - 1e-4]
        con = np.array(g + h)
        con[con<0] = 0

        return f, con
    
    def C04(self, x):
        shift_data = np.loadtxt("/home/yongcun/work/optimize/ec/problems/data/shift_data_4.txt")
        o = shift_data[:self.dim]
        z = x-o

        f = np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)
        g = [-np.sum(z*np.sin(2*z)), np.sum(z*np.sin(z))]
        h = [0]
        con = np.array(g + h)
        con[con<0] = 0

        return f, con
    
    def C05(self, x):
        shift_data = np.loadtxt("/home/yongcun/work/optimize/ec/problems/data/shift_data_5.txt")
        o = shift_data[:self.dim]
        z = x-o
        M1 = np.loadtxt(f"/home/yongcun/work/optimize/ec/problems/data/M1_5_D{self.dim}.txt")
        M2 = np.loadtxt(f"/home/yongcun/work/optimize/ec/problems/data/M2_5_D{self.dim}.txt")
        y = np.dot(M1, z.reshape(-1,1)).squeeze()
        w = np.dot(M2, z.reshape(-1,1)).squeeze()
        f =  np.sum(100*(z[:-1]**2 - z[1:])**2 + (z[:-1]-1)**2)
        g = [np.sum(y**2 - 50*np.cos(2*np.pi*y) - 40), np.sum(w**2 - 50*np.cos(2*np.pi*w) - 40)]
        h = [0]
        con = np.array(g + h)
        con[con<0] = 0

        return f, con
    

if __name__ == "__main__":
    x = np.random.uniform(-10, 10, 10)
    
    problem = CEC2017("C05", 10) 

    print(problem.evaluate(x))
    


