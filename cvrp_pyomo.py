import time
import pyomo.environ as pyo
import scipy.spatial.distance as spd

from problems.load_cvrp import CVRP
from problems.plot_cvrp import plot_path as ppx
from problems.plot_tsp import plot_path as pp1


class CVRPOpt:
    def __init__(self, prob):
        self.n_trucks = prob.n_trucks
        self.n_nodes = prob.n_nodes
        self.capacity = prob.capacity
        self.graph = prob.graph
        self.demands = prob.demands

        # 增加一个虚拟终点车场
        self.n_nodes = self.n_nodes+1
        self.demands[self.n_nodes] = self.demands[1]
        self.graph[self.n_nodes] = self.graph[1]

        self.dis_matrix = spd.squareform(spd.pdist(list(prob.graph.values())))

    def __obj_func(self, model):
        return pyo.quicksum(model.x[i,j,k] * self.dis_matrix[i-1, j-1] for i,j,k in model.xset)
    
    def __rule_con1(self, model, I):
        return pyo.quicksum(model.x[I,j,k] for i,j,k in model.xset if i == I) == 1
    
    def __rule_con2_1(self, model, K):
        return  pyo.quicksum(model.x[1,j,K] for i,j,k in model.xset if i == 1 and k == K) == 1
    def __rule_con2_2(self, model, K):
        return pyo.quicksum(model.x[i,self.n_nodes,K] for i,j,k in model.xset if j == self.n_nodes and k == K) == 1
    
    def __rule_con3(self, model, H, K):
        return pyo.quicksum(model.x[i, H, K] for i,j,k in model.xset if k==K and j==H) == pyo.quicksum(model.x[H, j, K] for i,j,k in model.xset if k==K and i==H)

    def __rule_con5(self, model, K):
        return pyo.quicksum(model.x[i, j, K] * self.demands[i] for i,j,k in model.xset if k == K) <= self.capacity
    
    def create_model(self):
        
        self.model = pyo.ConcreteModel()

        self.model.V = pyo.Set(initialize=range(1, self.n_nodes+1))
        self.model.C = self.model.V - [1, self.n_nodes]
        self.model.K = pyo.Set(initialize=list(range(1, self.n_trucks+1)))

        # 决策变量
        X_set = []
        for k in self.model.K:
            for i in self.model.V-{self.n_nodes}:
                for j in self.model.V:
                    if i!=j:
                        X_set.append((i, j, k))
        self.model.xset = pyo.Set(initialize=X_set)
        self.model.x = pyo.Var(self.model.xset, within=pyo.Binary)
        self.model.u = pyo.Var(self.model.V, self.model.K, within=pyo.NonNegativeReals)

        # 目标函数
        self.model.obj = pyo.Objective(rule=self.__obj_func, sense=pyo.minimize)

        # 约束条件el.V, self.mo
        self.model.con1 = pyo.Constraint(self.model.C, rule=self.__rule_con1) 
        
        self.model.con2_1 = pyo.Constraint(self.model.K, rule=self.__rule_con2_1)
        self.model.con2_2 = pyo.Constraint(self.model.K, rule=self.__rule_con2_2)

        self.model.con3 = pyo.Constraint(self.model.C, self.model.K, rule=self.__rule_con3)

        # 3. 容量和破子圈约束
        self.model.con4 = pyo.ConstraintList()
        for i in self.model.V - {self.n_nodes}:
            for j in self.model.V - {1}:
                if i != j:
                    for k in self.model.K:
                            expr = self.model.u[i,k] - self.model.u[j,k] + self.n_nodes * self.model.x[i,j,k] <= self.n_nodes - 1
                            self.model.con4.add(expr=expr)
    
        self.model.con5 = pyo.Constraint(self.model.K, rule=self.__rule_con5)

    def print_model(self):
        self.model.pprint()

    def solve_model(self, solver_name,TIME_LIMIT):
        start = time.time()
        solver = pyo.SolverFactory(solver_name)
        if 'cplex' in solver_name:
            solver.options['timelimit'] = TIME_LIMIT
        elif 'glpk' in solver_name:         
            solver.options['tmlim'] = TIME_LIMIT
        elif 'gurobi' in solver_name:           
            solver.options['TimeLimit'] = TIME_LIMIT
        elif 'xpress' in solver_name:
            solver.options['maxtime'] = TIME_LIMIT 
        results = solver.solve(self.model, timelimit=TIME_LIMIT, tee=True)
        end = time.time()

        arcs = []
        try:
            print(f"最优解是：{pyo.value(self.model.obj):.2f}, 运行时间： {end-start:.2f} s")
            for idx in self.model.x:
                if pyo.value(self.model.x[idx]) > 0.5:
                    arcs.append(idx)
            print("已成功获取路径信息")
            return arcs
        except:
            print("模型不可行")
            return []

class CVRPOpt2:
    def __init__(self, prob):
        self.n_trucks = prob.n_trucks
        self.n_nodes = prob.n_nodes
        self.capacity = prob.capacity
        self.graph = prob.graph
        self.dis_matrix = spd.squareform(spd.pdist(list(prob.graph.values())))
        self.demands = prob.demands

    def create_model(self):
        #create model
        self.model = pyo.ConcreteModel("CVRP")
        #create set
        self.model.N = pyo.Set(initialize=self.demands.keys())
        self.model.C = self.model.N - {1}

        #create decision variable vtype = Binary
        self.model.x = pyo.Var(self.model.N,self.model.N,within=pyo.Binary,initialize=0)

        #flow Constraint any customer is visited only once
        def inflow(model,i):
            return pyo.quicksum(self.model.x[i,j] for j in self.model.N if j != i) == 1
        #flow balance Constraint
        def flowBlance(model,i):
            return pyo.quicksum(self.model.x[i,j] for j in self.model.N if j !=i) ==\
                        pyo.quicksum(self.model.x[j,i] for j in self.model.N if j !=i)
        
        self.model.c1 = pyo.Constraint(self.model.C,rule=inflow)
        self.model.c2 = pyo.Constraint(self.model.C,rule=flowBlance)

        #Constraints of Depot
        self.model.c3 = pyo.Constraint(expr=pyo.quicksum(self.model.x[1,j] for j in self.model.C)>=1)
        self.model.c4 = pyo.Constraint(expr=pyo.quicksum(self.model.x[1,j] for j in self.model.C)<=\
                                            self.n_trucks)
        self.model.c5 = pyo.Constraint(expr=pyo.quicksum(self.model.x[1,j] for j in self.model.C)==\
                                            pyo.quicksum(self.model.x[j,1] for j in self.model.C))

        #Constraints Capacity
        self.model.q = pyo.Var(self.model.N,bounds=[0,self.capacity],\
                               within=pyo.NonNegativeReals)
        
        M = self.capacity
        self.model.c6 = pyo.ConstraintList()
        for i in self.model.N:
            for j in self.model.C:
                if i != j:
                    expr = self.model.q[i] + self.demands[j] - (1-self.model.x[i,j])*M
                    self.model.c6.add(expr=expr<=self.model.q[j])

        #Objective
        def objRule(model):
            return pyo.quicksum(self.dis_matrix[i-1,j-1] * self.model.x[i,j] \
                                for i in self.model.N for j in self.model.N)
        self.model.obj = pyo.Objective(rule=objRule,sense=pyo.minimize)

    def solve_model(self, solver_name,TIME_LIMIT):
        start = time.time()
        solver = pyo.SolverFactory(solver_name)
        if 'cplex' in solver_name:
            solver.options['timelimit'] = TIME_LIMIT
        elif 'glpk' in solver_name:         
            solver.options['tmlim'] = TIME_LIMIT
        elif 'gurobi' in solver_name:           
            solver.options['TimeLimit'] = TIME_LIMIT
        elif 'xpress' in solver_name:
            solver.options['maxtime'] = TIME_LIMIT 
        results = solver.solve(self.model, timelimit=TIME_LIMIT, tee=True)
        end = time.time()
    
        arcs = []
        try:
            print(f"最优解是：{pyo.value(self.model.obj):.2f}, 运行时间： {end-start:.2f} s")
            for idx in self.model.x:
                if pyo.value(self.model.x[idx]) > 0.5:
                    arcs.append(idx)
            print("已成功获取路径信息")
            return arcs
        except:
            print("模型不可行")
            return []



if __name__ == "__main__":
    # 问题
    fpath = "/home/yongcun/work/optimize/ec/problems/CVRP/A-n32-k5.vrp"
    prob = CVRP(fpath)
    # print(prob.graph)
    
    # 优化
    opt = CVRPOpt(prob)
    opt.create_model()
    # opt.print_model()
    
    edges = opt.solve_model('gurobi_direct', TIME_LIMIT=300)  # gurobi_direct scip
    print(edges)
    # pp1(edges, prob.graph)
    ppx(edges, prob.graph)
    







