import numpy as np
import math
import pandas as pd
class TSP:
    def __init__(self, path):
        self.tmap = self.load(path)
        self.size = len(self.tmap)
        self.dist = self.getDist()

    #读取城市的x，y坐标
    def load(self, txt):
        f = open(txt)
        tmap=[]
        flag = 0
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                flag = 1
                continue
            if line == "EOF":
                break
            if flag:
                a = line.split()
                tmap.append((float(a[1]),float(a[2])))
        return tuple(tmap)

    #获取两个城市间的二维欧几里得距离
    def getDist(self):
        dist = np.zeros((self.size,self.size))
        for i in range(0,self.size):
            for j in range(0,self.size):
                if i == j:
                    # 相同城市不允许访问
                    dist[i][j] = np.inf
                else:
                    dist[i][j] = np.sqrt((self.tmap[i][0]-self.tmap[j][0])**2 + (self.tmap[i][1]-self.tmap[j][1])**2)
        return dist
    
    def getCost(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.dist[path[i]][path[i + 1]]
        distance += self.dist[path[-1]][path[0]]
        return distance


class CTSP:
    def __init__(self):
        original_cities = [['西宁', 101.74, 36.56],
                       ['兰州', 103.73, 36.03],
                       ['银川', 106.27, 38.47],
                       ['西安', 108.95, 34.27],
                       ['郑州', 113.65, 34.76],
                       ['济南', 117, 36.65],
                       ['石家庄', 114.48, 38.03],
                       ['太原', 112.53, 37.87],
                       ['呼和浩特', 111.65, 40.82],
                       ['北京', 116.407526, 39.90403],
                       ['天津', 117.200983, 39.084158],
                       ['沈阳', 123.38, 41.8],
                       ['长春', 125.35, 43.88],
                       ['哈尔滨', 126.63, 45.75],
                       ['上海', 121.473701, 31.230416],
                       ['杭州', 120.19, 30.26],
                       ['南京', 118.78, 32.04],
                       ['合肥', 117.27, 31.86],
                       ['武汉', 114.31, 30.52],
                       ['长沙', 113, 28.21],
                       ['南昌', 115.89, 28.68],
                       ['福州', 119.3, 26.08],
                       ['台北', 121.3, 25.03],
                       ['香港', 114.173355, 22.320048],
                       ['澳门', 113.54909, 22.198951],
                       ['广州', 113.23, 23.16],
                       ['海口', 110.35, 20.02],
                       ['南宁', 108.33, 22.84],
                       ['贵阳', 106.71, 26.57],
                       ['重庆', 106.551556, 29.563009],
                       ['成都', 104.06, 30.67],
                       ['昆明', 102.73, 25.04],
                       ['拉萨', 91.11, 29.97],
                       ['乌鲁木齐', 87.68, 43.77]]
        
        original_cities = pd.DataFrame(original_cities, columns=['城市', '经度', '纬度'])
        D = original_cities[['经度', '纬度']].values * math.pi / 180
        city_cnt = len(original_cities)
        dist_mat = np.zeros((city_cnt, city_cnt))
        for i in range(city_cnt):
            for j in range(city_cnt):
                if i == j:
                    # 相同城市不允许访问
                    dist_mat[i][j] = 1000000
                else:
                    # 单位：km
                    dist_mat[i][j] = 6378.14 * math.acos(
                        math.cos(D[i][1]) * math.cos(D[j][1]) * math.cos(D[i][0] - D[j][0]) +
                        math.sin(D[i][1]) * math.sin(D[j][1]))
        
        self.size = city_cnt
        self.dist = dist_mat

    def getCost(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.dist[path[i]][path[i + 1]]
        distance += self.dist[path[-1]][path[0]]
        return distance






if __name__ == "__main__":
    fpath = "/home/yongcun/work/optimize/ec/problems/TSP/ch130.txt"
    prob = TSP(fpath)

    print(prob.size)


