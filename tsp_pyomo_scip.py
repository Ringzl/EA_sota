import time
import pyomo.environ as pyo
from problems.load_tsp import TSP

from problems.plot_tsp import plot_path

class TspOpt:
    def __init__(self, cost_matrix, num_cities):
        self.cost_matrix = cost_matrix
        self.num_cities = num_cities
    
    @staticmethod
    def __obj_fun(model):
        return pyo.quicksum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M if i != j)

    @staticmethod
    def __rule_con1(model, M):
        return pyo.quicksum(model.x[i, M] for i in model.N if i != M) == 1
    
    @staticmethod
    def __rule_con2(model, N):
        return pyo.quicksum(model.x[N, j] for j in model.M if j != N) == 1
    
    @staticmethod
    def __rule_con3(model, i, j):
        if i != j:
            return model.u[i] - model.u[j] + model.x[i,j] * model.num_cities <= model.num_cities - 1
        else:
            return model.u[i] - model.u[i] == 0 

    def create_model(self):
        self.model = pyo.ConcreteModel()

        # 变量取值集合
        self.model.M = pyo.RangeSet(self.num_cities)
        self.model.N = pyo.RangeSet(self.num_cities)
        self.model.U = pyo.RangeSet(2, self.num_cities)

        # 参数: 路径权重
        self.model.c = pyo.Param(self.model.N, self.model.M, initialize=lambda model, i, j: self.cost_matrix[i-1][j-1])

        # 决策变量
        self.model.x = pyo.Var(self.model.N, self.model.M, within=pyo.Binary)

        self.model.num_cities = self.num_cities

        # 辅助变量
        self.model.u = pyo.Var(self.model.N, within=pyo.NonNegativeReals, bounds=[0, self.num_cities-1])

        # 目标函数
        self.model.obj = pyo.Objective(rule=self.__obj_fun, sense=pyo.minimize)

        # 约束
        # 只访问一次
        self.model.con1 = pyo.Constraint(self.model.N, rule=self.__rule_con1)
        self.model.con2 = pyo.Constraint(self.model.M, rule=self.__rule_con2)
        
        # 消除子回路
        self.model.con3 = pyo.Constraint(self.model.U, self.model.N, rule=self.__rule_con3)
    
    def print_model(self):
        self.model.pprint()

    def solve_model(self, solver_name):
        start = time.time()
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(self.model)
        end = time.time()
        # print(results)

        arcs = [] # 路径
        try:
            print(f"最优解是：{pyo.value(self.model.obj):.2f}, 运行时间： {end-start:.2f} s")
            for idx in self.model.x:
                if self.model.x[idx]() is not None and self.model.x[idx]() != 0:
                    arcs.append(idx)
            print("已成功获取路径信息")
            return arcs
        except:
            print("模型不可行")
    

if __name__ == "__main__":
    # # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # print(prob.size, prob.dist.shape)

    # # 优化
    # opt = TspOpt(prob.dist, prob.size)
    # opt.create_model()
    # edges = opt.solve_model('scip')  # gurobi_direct
    # plot_path(edges,  pos_dct)
    
   
    
    p_lst = [
        'berlin52', 'ch130', 'd198', 'd493',
        'd657', 'd1291'
    ]
    
    for p in p_lst:
        # 问题
        fpath = f"/home/yongcun/work/optimize/ec/problems/TSP/{p}.tsp"
        print(f"问题： {p}")
        prob = TSP(fpath)
        pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
        # print(prob.size, prob.dist.shape)

        # 优化
        opt = TspOpt(prob.dist, prob.size)
        opt.create_model()
        edges = opt.solve_model('gurobi_direct')  # scip
        plot_path(edges,  pos_dct)
    



