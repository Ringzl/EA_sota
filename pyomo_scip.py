import pyomo.environ as pyo

class TSP:
    def __init__(self, cost_matrix, num_cities):
        self.cost_matrix = cost_matrix
        self.num_cities = num_cities
    
    @staticmethod
    def __obj_fun(model):
        return pyo.quicksum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M)

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
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(self.model)
        print(results)
    

if __name__ == "__main__":
    pass



