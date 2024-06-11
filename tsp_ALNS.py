"""
对于一个邻域搜索算法，当其邻域大小随着输入数据的规模大小呈指数增长的时候，
那么我们就可以称该邻域搜索算法为超大规模邻域搜索算法(Very Large Scale Neighborhood Search Algorithm, VLSNA)
邻域是由destroy和repair方法隐式定义:
destroy方法会破坏当前解的一部分, 而后repair方法会对被破坏的解进行重建

ALNS会为每个destroy和repair方法分配一个权重,
通过该权重从而控制每个destroy和repair方法在搜索期间使用的频率。

ALNS通过使用多种destroy和repair方法, 
然后再根据这些destroy和repair方法生成的解的质量, 
选择那些表现好的destroy和repair方法, 再次生成邻域进行搜索。
"""

import random
import numpy as np
import copy
import time

from problems.load_tsp import TSP
from problems.plot_tsp import plot_path

class ALNS:
    def __init__(self, prob, max_iter):
        self.prob = prob
        self.max_iter = max_iter
        self.size = self.prob.size
        self.dist = self.prob.dist

        # 模拟退火温度
        self.T = 100
        # 降温速度
        self.a = 0.97

        # destroy的城市数量
        self.destroy_city_cnt = int(self.size * 0.1)
        # destroy和repair的挥发系数
        self.lambda_rate = 0.5

        self.x = []
        self.cost = float("inf")


    # 随机删除N个城市
    def random_destroy(self, x):
        new_x = copy.deepcopy(x)
        removed_cities = []

        # 随机选择N个城市，并删除
        removed_index = random.sample(range(0, len(x)), self.destroy_city_cnt)
        for i in removed_index:
            removed_cities.append(new_x[i])
            x.remove(new_x[i])
        return removed_cities


    # 删除距离最大的N个城市
    def max_n_destroy(self, x):
        new_x = copy.deepcopy(x)
        removed_cities = []

        # 计算顺序距离并排序
        dis = []
        for i in range(len(new_x) - 1):
            dis.append(self.dist[new_x[i]][new_x[i + 1]])
        dis.append(self.dist[new_x[-1]][new_x[0]])
        sorted_index = np.argsort(np.array(dis))

        # 删除最大的N个城市
        for i in range(self.destroy_city_cnt):
            removed_cities.append(new_x[sorted_index[-1 - i]])
            x.remove(new_x[sorted_index[-1 - i]])

        return removed_cities

    # 随机删除连续的N个城市
    def continue_n_destroy(self, x):

        new_x = copy.deepcopy(x)
        removed_cities = []

        # 随机选择N个城市，并删除
        removed_index = random.sample(range(0, len(x)-self.destroy_city_cnt), 1)[0]
        for i in range(removed_index + self.destroy_city_cnt, removed_index, -1):
            removed_cities.append(new_x[i])
            x.remove(new_x[i])
        return removed_cities

    # destroy操作
    def destroy(self, flag, x):
        # 三个destroy算子，第一个是随机删除N个城市，第二个是删除距离最大的N个城市，第三个是随机删除连续的N个城市
        removed_cities = []
        if flag == 0:
            # 随机删除N个城市
            removed_cities = self.random_destroy(x)
        elif flag == 1:
            # 删除距离最大的N个城市
            removed_cities = self.max_n_destroy(x)
        elif flag == 2:
            # 随机删除连续的N个城市
            removed_cities = self.continue_n_destroy(x)
        return removed_cities
    

    # 随机插入
    def random_insert(self, x, removed_cities):
        insert_index = random.sample(range(0, len(x)), len(removed_cities))
        for i in range(len(insert_index)):
            x.insert(insert_index[i], removed_cities[i])

    # 贪心插入
    def greedy_insert(self, x, removed_cities):
        dis = float('inf')
        insert_index = -1

        for i in range(len(removed_cities)):
            # 寻找插入后的最小总距离
            for j in range(len(x) + 1):
                new_x = copy.deepcopy(x)
                new_x.insert(j, removed_cities[i])
                cost = self.prob.getCost(new_x)
                if cost < dis:
                    dis = cost
                    insert_index = j

            # 最小位置处插入
            x.insert(insert_index, removed_cities[i])
            dis = float('inf')

    # repair操作
    def repair(self, flag, x, removed_cities):
        # 两个repair算子，第一个是随机插入，第二个贪心插入
        if flag == 0:
            self.random_insert(x, removed_cities)
        elif flag == 1:
            self.greedy_insert(x, removed_cities)

    # 选择destroy算子
    def select_and_destroy(self, destroy_w, x):
        # 轮盘赌逻辑选择算子
        prob = destroy_w / np.array(destroy_w).sum()
        seq = [i for i in range(len(destroy_w))]
        destroy_operator = np.random.choice(seq, size=1, p=prob)[0]

        # destroy操作
        removed_cities = self.destroy(destroy_operator, x)

        return x, removed_cities, destroy_operator


    # 选择repair算子
    def select_and_repair(self, repair_w, x, removed_cities):
        # # 轮盘赌逻辑选择算子
        prob = repair_w / np.array(repair_w).sum()
        seq = [i for i in range(len(repair_w))]
        repair_operator = np.random.choice(seq, size=1, p=prob)[0]

        # repair操作：此处设定repair_operator=1，即只使用贪心策略
        self.repair(1, x, removed_cities)

        return x, repair_operator


    def run(self):
        # destroy算子的初始权重
        destroy_w = [1, 1, 1]
        # repair算子的初始权重
        repair_w = [1, 1]
        # destroy算子的使用次数
        destroy_cnt = [0, 0, 0]
        # repair算子的使用次数
        repair_cnt = [0, 0]
        # destroy算子的初始得分
        destroy_score = [1, 1, 1]
        # repair算子的初始得分
        repair_score = [1, 1]

         # 当前解，第一代，贪心策略生成
        removed_cities = [i for i in range(self.size)]
        
        self.repair(1, self.x, removed_cities)
        self.cost = self.prob.getCost(self.x)
        
        # 历史最优解解，第一代和当前解相同，注意是深拷贝，此后有变化不影响x，也不会因x的变化而被影响
        self.history_best_x = copy.deepcopy(self.x)
        self.history_best_cost = self.cost

        cur_iter = 0
        while cur_iter < self.max_iter:
            
            # destroy算子
            destroyed_x, remove, destroy_operator_index = self.select_and_destroy(destroy_w, self.x)
            destroy_cnt[destroy_operator_index] += 1

            # repair算子
            new_x, repair_operator_index = self.select_and_repair(repair_w, destroyed_x, remove)
            repair_cnt[repair_operator_index] += 1

            cost_new = self.prob.getCost(new_x)
            if cost_new <= self.cost:
                # 测试解更优，更新当前解
                self.x = new_x
                self.cost = cost_new
                if cost_new <= self.history_best_cost:
                    # 测试解为历史最优解，更新历史最优解，并设置最高的算子得分
                    self.history_best_x = new_x
                    self.history_best_cost = cost_new
                    destroy_score[destroy_operator_index] = 1.5
                    repair_score[repair_operator_index] = 1.5
                else:
                    # 测试解不是历史最优解，但优于当前解，设置第二高的算子得分
                    destroy_score[destroy_operator_index] = 1.2
                    repair_score[repair_operator_index] = 1.2
            else:
                if np.random.random() < np.exp((self.cost - cost_new)) / self.T:
                    # 当前解优于测试解，但满足模拟退火逻辑，依然更新当前解，设置第三高的算子得分
                    self.x = copy.deepcopy(new_x)
                    destroy_score[destroy_operator_index] = 0.8
                    repair_score[repair_operator_index] = 0.8
                else:
                    # 当前解优于测试解，也不满足模拟退火逻辑，不更新当前解，设置最低的算子得分
                    destroy_score[destroy_operator_index] = 0.5
                    repair_score[repair_operator_index] = 0.5

            # 更新destroy算子的权重
            destroy_w[destroy_operator_index] = \
                destroy_w[destroy_operator_index] * self.lambda_rate + \
                (1 - self.lambda_rate) * destroy_score[destroy_operator_index] / destroy_cnt[destroy_operator_index]
            # 更新repair算子的权重
            repair_w[repair_operator_index] = \
                repair_w[repair_operator_index] * self.lambda_rate + \
                (1 - self.lambda_rate) * repair_score[repair_operator_index] / repair_cnt[repair_operator_index]
            # 降低温度
            self.T = self.a * self.T

            # 结束一轮迭代，重置模拟退火初始温度
            cur_iter += 1
            # print(
            #     'cur_iter: {}, best_f: {}'.format(cur_iter, self.history_best_cost))

        return self.history_best_x, self.history_best_cost



if __name__ == "__main__":

    # 问题
    # fpath = "/home/yongcun/work/optimize/ec/problems/TSP/berlin52.tsp"
    # prob = TSP(fpath)
    # pos_dct =  dict(zip(list(range(prob.size)), prob.tmap))
    # # 优化
    # start = time.time()
    # alns = ALNS(prob=prob, max_iter=100)
    # route, cost = alns.run()
    # end = time.time()
    # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
    # edges = list(zip(route[:-1], route[1:]))
    # edges.append((route[-1], route[0]))
    # plot_path(edges, pos_dct)


    p_lst = [
        'berlin52', 'ch130', 'd198', 
        'd493',
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
            alns = ALNS(prob=prob, max_iter=100)
            route, cost = alns.run()
            end = time.time()
            c_lst.append(cost)
            t_lst.append(end-start)
            # print(f"最优解是：{cost:.2f}, 运行时间： {end-start:.2f} s")
            edges = list(zip(route[:-1], route[1:]))
            edges.append((route[-1], route[0]))
            # plot_path(edges,  pos_dct)
        
        print((f"最优解是：{np.mean(c_lst):.2f}±{np.std(c_lst):.2f}, 运行时间： {np.mean(t_lst):.2f} s"))

    

    
            
