import random
import time
import numpy as np
from problems.load_tsp import TSP
from problems.plot_tsp import plot_path

class VNS:
    def __init__(self, prob, kmax):
        self.prob = prob
        self.kmax = kmax

        # 初始
        self.size = self.prob.size
        self.start = tuple([k for k in range(self.size)])
        self.visited = {}
        self.visited.update({self.start: self.prob.getCost(self.start)})

        self.route = self.start
        self.cost_min = self.visited[self.start]
        self.count = 0
    
    #在Local Search中使用VND方法进行搜索        
    def VND(self, path):
        l = 0
        cost_min = self.visited[path]
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

    def shaking(self, path):
        ini = self.visited[path]
        cnt = 0
        while True:
            pos1,pos2,pos3 = sorted(random.sample(range(0,self.size),3))
            path_ = path[pos1:pos2] + path[:pos1] + path[pos3:] + path[pos2:pos3]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            cnt+=1
            if ini >= cost:
                break
            elif cnt > 100:
                path_ = path
                cost = ini
                break
        return path_
    
    #反转一段区间，获取新邻域
    def getNei_rev(self, path):
        cost_min = self.visited[path]
        cnt = 0
        while True:
            i,j = sorted(random.sample(range(1,self.size-1),2))
            path_ = path[:i] + path[i:j+1][::-1] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            cnt+=1
            if cost < cost_min:
                cost_min = cost
                break
            elif cnt > 1000:
                path_ = path
                break
        return path_,cost_min

    #交换两个城市，获取新邻域
    def getNei_exc(self, path):
        cost_min = self.visited[path]
        cnt = 0
        while True:
            i,j = sorted(random.sample(range(1, self.size-1),2))
            path_ = path[:i] + path[j:j+1] + path[i+1:j] + path[i:i+1] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            cnt+=1
            if cost < cost_min:
                cost_min = cost
                break
            elif cnt > 1000:
                path_ = path
                break
        return path_,cost_min
        
    #随机挑选两个城市插入序列头部，获取新邻域
    def getNei_ins(self, path):
        cost_min = self.visited[path]
        cnt = 0
        while True:
            i,j = sorted(random.sample(range(1,self.size-1),2))
            path_ = path[i:i+1] + path[j:j+1] + path[:i] + path[i+1:j] + path[j+1:]
            if path_ not in self.visited:
                cost = self.prob.getCost(path_)
                self.visited.update({path_:cost})
            else:
                cost = self.visited[path_]
            cnt+=1
            if cost < cost_min:
                cost_min = cost
                break
            elif cnt > 1000:
                path_ = path
                break
        return path_, cost_min


    def run(self):
        k = 0
        while k < self.kmax:
            # 扰动后进行变邻域操作
            route_nei, cost = self.VND(self.shaking(self.route))
            self.count += 1

            if cost < self.cost_min:
                self.route = route_nei
                self.cost_min = cost
                k = 0
            else:
                k += 1
            # print(self.cost_min)

        return self.route, self.cost_min
    
if __name__ == "__main__":
     # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # 优化
    # start = time.time()
    # vns = VNS(prob=prob, kmax=50)
    # route, cost = vns.run()
    # end = time.time()
    # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
    # edges = list(zip(route[:-1], route[1:]))
    # edges.append((route[-1], route[0]))
    # plot_path(edges, pos_dct)


    p_lst = [
        # 'berlin52', 'ch130', 'd198', 'd493',
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
            vns = VNS(prob=prob, kmax=50)
            route, cost = vns.run()
            end = time.time()
            c_lst.append(cost)
            t_lst.append(end-start)
            # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
            edges = list(zip(route[:-1], route[1:]))
            edges.append((route[-1], route[0]))
            # plot_path(edges,  pos_dct)
        
        print((f"最优解是：{np.mean(c_lst):.2f}±{np.std(c_lst):.2f}, 运行时间： {np.mean(t_lst):.2f} s"))

    
            

        

