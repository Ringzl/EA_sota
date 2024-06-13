import random
import numpy as np
from problems.load_tsp import TSP
from problems.plot_tsp import plot_path

import time

class SA:
    def __init__(self, prob, kmax=100):
        self.prob = prob
        self.kmax = kmax

        self.t0 = 500000
        self.t_end = 0.00001

        self.size = self.prob.size
        self.start = tuple([k for k in range(self.size)])
        self.visited = {}
        self.visited.update({self.start: self.prob.getCost(self.start)})

        self.route = self.start
        self.cost_min = self.visited[self.start]

    #反转一段区间，获取新邻域
    def getNei_rev(self, path):
        cost_min = np.inf
        for cnt in range(100):
            i,j = sorted(random.sample(range(1, self.size-1),2))
            path_ = path[:i] + path[i:j+1][::-1] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            if cost < self.visited[path]:
                cost_min = cost
                p = path_
                break
            if cost < cost_min:
                cost_min = cost
                p = path_
        return p,cost_min
    
    #交换两个城市，获取新邻域
    def getNei_exc(self, path):
        cost_min = np.inf
        for cnt in range(100):
            i,j = sorted(random.sample(range(1,self.size-1),2))
            path_ = path[:i] + path[j:j+1] + path[i+1:j] + path[i:i+1] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            if cost < self.visited[path]:
                cost_min = cost
                p = path_
                break
            if cost < cost_min:
                cost_min = cost
            p = path_
        return p,cost_min
    
    #随机挑选两个城市插入序列头部，获取新邻域
    def getNei_ins(self, path):
        cost_min = np.inf
        for cnt in range(100):
            i,j = sorted(random.sample(range(1,self.size-1),2))
            path_ = path[i:i+1] + path[j:j+1] + path[:i] + path[i+1:j] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            if cost < self.visited[path]:
                cost_min = cost
                p = path_
                break
            if cost < cost_min:
                cost_min = cost
                p = path_
        return p, cost_min
    
    #在Local Search中使用VND方法进行搜索  
    def VND(self, path):
        path, cost_min = self.getNei_rev(path)
        l = 1
        while l < 3:
            if l == 0:
                path_,cost = self.getNei_rev(path)
            elif l == 1:
                path_,cost = self.getNei_exc(path)
            elif l == 2:
                path_,cost = self.getNei_ins(path)
            if cost < cost_min:
                path = path_
                cost_min = cost
                l = 0
            else:
                l+=1
        return path, cost_min  
    
    def run(self):
        temp = self.start
        result = [temp, self.cost_min]
        t = self.t0
        while t > self.t_end:
            for k in range(self.kmax):
                path_nei,cost = self.VND(temp) #进行变邻域操作
                
                #判断是否接受该解
                if cost < self.cost_min or random.random() < np.exp(-((cost-self.cost_min)/t*k)):
                    temp = path_nei
                    self.cost_min = cost

                    if cost <= result[1]:
                        result = [path_nei, cost]
            t /= k + 1
        return result[0], result[1] 
                
                    
if __name__ == "__main__":
    # # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # 优化
    # start = time.time()
    # sa = SA(prob)
    # route, cost = sa.run()
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
            sa = SA(prob)
            route, cost = sa.run()
            end = time.time()
            c_lst.append(cost)
            t_lst.append(end-start)
            # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
            edges = list(zip(route[:-1], route[1:]))
            edges.append((route[-1], route[0]))
            # plot_path(edges,  pos_dct)
        
        print((f"最优解是：{np.mean(c_lst):.2f}±{np.std(c_lst):.2f}, 运行时间： {np.mean(t_lst):.2f} s"))
        



    




