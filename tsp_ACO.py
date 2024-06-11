import numpy as np
from problems.load_tsp import TSP
from problems.plot_tsp import plot_path
import time

class ACO:
    def __init__(self, prob, ant_cnt=100, max_iter=200):

        self.prob = prob

        # 城市数量
        self.city_cnt = self.prob.size
        # 蚂蚁数量
        self.ant_cnt = ant_cnt
        # 迭代次数
        self.max_iter = max_iter
        # 信息素权重系数
        self.alpha = 1
        # 启发信息权重系数
        self.beta = 2
        # 信息素挥发速度
        self.rho = 0.1

        # 启发信息，距离倒数
        self.distance = self.prob.dist.copy()
        for i in range(self.city_cnt):
            self.distance[i][i] = 1000000
        self.eta = 1 / self.distance

        # 信息素矩阵
        self.tau = np.ones((self.city_cnt, self.city_cnt))  

        # 每一代最优解
        self.generation_best_y = []
        self.generation_best_x = []

        # 种群
        self.pop = np.zeros((ant_cnt, self.city_cnt)).astype(int)

    
    def run(self):
        # 循环迭代
        for i in range(self.max_iter):
            # 城市转移概率
            prob_matrix = (self.tau ** self.alpha) * (self.eta ** self.beta)

            # TSP距离
            y = np.zeros(self.ant_cnt)

            # 依次遍历每只蚂蚁
            for j in range(self.ant_cnt):
                # 设置TSP初始点为0
                self.pop[j, 0] = 0

                # 选择后续城市
                for k in range(self.city_cnt - 1):
                    # 已访问城市
                    visit = set(self.pop[j, :k + 1])
                    # 未访问城市
                    un_visit = list(set(range(self.city_cnt)) - visit)
                    # 未访问城市转移概率归一化
                    prob = prob_matrix[self.pop[j, k], un_visit]
                    prob = prob / prob.sum()
                    # 轮盘赌策略选择下个城市
                    next_point = np.random.choice(un_visit, size=1, p=prob)[0]
                    # 添加被选择的城市
                    self.pop[j, k + 1] = next_point
                    # 更新TSP距离
                    y[j] += self.distance[self.pop[j, k], self.pop[j, k + 1]]
                # 更新TSP距离：最后一个城市->第0个城市
                y[j] += self.distance[self.pop[j, -1], 0]

            # 保存当前代最优解
            best_index = y.argmin()
            self.generation_best_x.append(self.pop[best_index, :])
            self.generation_best_y.append(y[best_index])

            # 计算信息素改变量，ACS模型，Q=1
            delta_tau = np.zeros((self.city_cnt, self.city_cnt))
            for j in range(self.ant_cnt):
                for k in range(self.city_cnt - 1):
                    delta_tau[self.pop[j, k],self.pop[j, k + 1]] += 1 / y[j]
                delta_tau[self.pop[j, self.city_cnt - 1], self.pop[j, 0]] += 1 / y[j]

            # 信息素更新
            self.tau = (1 - self.rho) * self.tau + delta_tau

            # print('iter: {}, best_f: {}'.format(i, self.generation_best_y[-1]))

        # 最优解位置
        best_generation_index = np.array(self.generation_best_y).argmin()

        return self.generation_best_x[best_generation_index], self.generation_best_y[best_generation_index]

if __name__ == "__main__":
    # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # 优化
    # start = time.time()
    # aco = ACO(prob=prob, ant_cnt=50, max_iter=100)
    # route, cost = aco.run()
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
            aco = ACO(prob=prob, ant_cnt=50, max_iter=100)
            route, cost = aco.run()
            end = time.time()
            c_lst.append(cost)
            t_lst.append(end-start)
            # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
            edges = list(zip(route[:-1], route[1:]))
            edges.append((route[-1], route[0]))
            # plot_path(edges,  pos_dct)
        
        print((f"最优解是：{np.mean(c_lst):.2f}±{np.std(c_lst):.2f}, 运行时间： {np.mean(t_lst):.2f} s"))



