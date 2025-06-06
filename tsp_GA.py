import random
import numpy as np
from problems.load_tsp import TSP
from problems.plot_tsp import plot_path

import time
class GA:
    def __init__(self, prob, popsize, kmax):
        self.prob = prob
        self.popsize = popsize
        self.kmax = kmax

        # 初始
        self.size = self.prob.size
        self.start = [tuple(random.sample(range(self.size),self.size)) for m in range(self.popsize)]
        self.visited = {key:self.prob.getCost(key) for key in self.start}

    def PMX(self, i,j):
        s,t = sorted(random.sample(range(1,self.size),2))
        next_i = list(i[:s] + j[s:t] + i[t:])
        next_j = list(j[:s] + i[s:t] + j[t:])
        #建立映射表
        mapped_i = {next_i[k]:next_j[k] for k in range(s,t)}
        mapped_j = {next_j[k]:next_i[k] for k in range(s,t)}
        #判断是否满足解的条件（每个城市皆访问一次）
        while len(set(next_i)) != len(next_i): 
            for k in range(self.size):
                if k < t and k >= s:
                    continue
                while next_i[k] in j[s:t]:
                    next_i[k] = mapped_i[next_i[k]]
        while len(set(next_j)) != len(next_j):
            for k in range(self.size):
                if k < t and k >= s:
                    continue
                while next_j[k] in i[s:t]:
                    next_j[k] = mapped_j[next_j[k]]
        next_i = tuple(next_i)
        next_j = tuple(next_j)
        if next_i not in self.visited:
            self.visited.update({next_i:self.prob.getCost(next_i)})
        if next_j not in self.visited:
            self.visited.update({next_j:self.prob.getCost(next_j)})
        return next_i,next_j
    
    def mutation(self, path):
        min = float('inf')
        for cnt in range(100):
            i,j = sorted(random.sample(range(1,self.size-1),2))
            path_ = path[:i] + path[i:j+1][::-1] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            if cost < self.visited[path]:
                min = cost
                p = path_
                break
            if cost < min:
                min = cost
                p = path_
        return p
    

    def run(self):
        temp = self.start
        for k in range(self.kmax):
            count = 0
            flag = 0

            children = [] #存储此代交叉、变异产生的子种群
            #加入当前种群中的最优解，使得下一代种群的最优解一定不会劣于当前种群最优解
            children.append(temp[0]) 
            
            for l in range(self.popsize):
                while True:
                    cur = sorted(temp[:], key=lambda x:self.visited[x])[0]
                    i = random.randrange(self.popsize)
                    count+=1
                    if temp[i] != cur:
                        break
                    if count > 100000:
                        flag = 1
                        break
                if flag == 0:
                    a,b = self.PMX(temp[i],cur) #使用PMX交叉操作
                    children.append(a)
                    children.append(b)
            for l in range(self.popsize):
                i = random.randrange(self.popsize)
                children.append(self.mutation(temp[i])) #使用反转作为变异操作
            temp = sorted(children[:], key=lambda x:self.visited[x])[:self.popsize] #选取子代中最优的前M个解
            
            # print(k,self.visited[temp[0]])
        return temp[0], self.visited[temp[0]]


if __name__ == "__main__":
    # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # 优化
    # start = time.time()
    # ga = GA(prob=prob, popsize=50,  kmax=100)
    # route, cost = ga.run()
    # end = time.time()
    # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
    # edges = list(zip(route[:-1], route[1:]))
    # edges.append((route[-1], route[0]))
    # plot_path(edges, pos_dct)


    p_lst = [
        'berlin52', 'ch130', 'd198', 'd493',
        'd657', 
        # 'd1291'
    ]
    M = 5
    for p in p_lst:
        # 问题
        fpath = f"/home/yongcun/work/optimize/ec/problems/TSP/{p}.tsp"
        print(f"问题： {p}")
        prob = TSP(fpath)
        pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
        # print(prob.size, prob.dist.shape)

        c_lst, t_lst = [], []
        for t in range(M):
            # 优化
            start = time.time()
            ga = GA(prob=prob, popsize=50,  kmax=100)
            route, cost = ga.run()
            end = time.time()
            c_lst.append(cost)
            t_lst.append(end-start)
            # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
            edges = list(zip(route[:-1], route[1:]))
            edges.append((route[-1], route[0]))
            # plot_path(edges,  pos_dct)
        
        print((f"最优解是：{np.mean(c_lst):.2f}±{np.std(c_lst):.2f}, 运行时间： {np.mean(t_lst):.2f} s"))

    
