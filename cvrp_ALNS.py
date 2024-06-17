
import re
import time
import numpy as np
from random import choices
from random import choice, randint, random
from copy import deepcopy
import matplotlib.pyplot as plt

import os
class Config:
    '''
    配置
    '''
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    D = 5
    ITERATION = 30000
    DESTROY_RATIO = 0.1
    SEGMENT = 300
    PHI_1 = 10
    PHI_2 = 6
    PHI_3 = 1
    # reaction factor
    N = 1
    # normalization factor
    V = 1
    # flat deviation
    DEVIATION = 30
    # broaden flat deviation
    BIG_DEVIATION = 100

'''
节点
'''
class Node:
    """
    :param id: ID cua khach hang
    :param x: Hoanh do khach hang
    :param y: Tung do khach hang
    :param demand: Nhu cau khoi luong hang hoa cua khach hang
    Ham xay dung cac thuoc tinh da neu o tren cua khach hang
    """
    def __init__(self, id, x, y, demand):
        # id of the customers
        self.id = int(id)
        # Coordinate of the customers
        self.x = np.array(int(x))
        self.y = np.array(int(y))
        # Demand of customers
        self.demand = int(demand)
    def __str__(self):
        return f'Customer: ID = {self.id}\t (x, y) = {"%2d" %self.x} {"%2d" %self.y}\t\t demand = {self.demand}\n'
    def __repr__(self):
        return str(self)
class Solution:
    """
    :param routes: Cac tuyen xe
    :param instance: Mo ta cua bai toan

    Ham xay dung loi giai bao gom:
        - instance (Instance): Mo ta bai toan
        - routes (list): Danh sach cac tuyen xe
        - extracted_node (list): Danh sach cac khach hang bi loai boi destroy operator
        - totalCost (int): Tong chi phi cua nghiem hien tai
    """
    def __init__(self, routes, instance):
        self.instance = instance
        self.routes = routes
        self.extracted_node = []
        self.totalCost = 0
        self.update_total_cost()

    def update_total_cost(self):
        self.totalCost = 0
        for route in self.routes:
            self.totalCost += route.cost

    def __str__(self):
        return f'Solution: Total cost = {"%.2f" %self.totalCost}, Routes = \n{self.routes})'
"""
路径
"""
class Route:
    """
    :param instance: Mo ta bai toan

    Ham xay dung tuyen duong danh cho xe
        - depot (Node): Diem do
        - route (list): Tuyen duong hien tai
        - cost (int): Chi phi van tai (tinh theo quang duong)
        - load (int): Luong hang hoa tren xe
        - max_load (int) luong hang hoa duoc phep toi da tren xe
        - distance_matrix (2D list): luu ma tran khoang cach giua cac khach hang
    """
    def __init__(self, instance):
        self.depot = instance.depot
        self.route = [instance.depot]
        self.cost = 0
        self.load = 0
        self.max_load = instance.capacity
        self.distance_matrix = instance.distance_matrix

    def last_node(self):
        return self.route[len(self.route) - 1]

    def can_insert(self, node):
        return node.demand + self.load <= self.max_load

    def insert(self, node, position=-1):
        if self.can_insert(node):
            self.load += node.demand
            if position != -1:
                #print(position)
                #print(self)
                self.cost = self.cost + self.distance_matrix[self.route[position].id][node.id] + self.distance_matrix[self.route[position + 1].id][node.id] - self.distance_matrix[self.route[position].id][self.route[position + 1].id]
                self.route.insert(position + 1, node)
                #print(self)
            else:
                self.cost += self.distance_matrix[self.last_node().id][node.id]
                self.route.append(node)

    def erase(self, customer_id):
        for i in range(0, len(self.route)):
            if self.route[i].id == customer_id:
                prev_customer = self.route[i-1].id
                removing_customer = self.route[i].id
                after_customer = self.route[i+1].id
                #copy = self.cost
                #copy2 = self.distance_matrix[prev_customer][removing_customer]
                #copy3 = self.distance_matrix[removing_customer][after_customer]
                #copy4 = self.distance_matrix[prev_customer][after_customer]
                self.cost = self.cost - self.distance_matrix[prev_customer][removing_customer] - self.distance_matrix[removing_customer][after_customer] + self.distance_matrix[prev_customer][after_customer]
                self.load -= self.route[i].demand
                self.route.remove(self.route[i])
                #if self.cost < 0:
                #    print(copy, copy2, copy3, copy4, self.cost)
                #    exit(999)
                break

    def finish_route(self):
        self.cost += self.distance_matrix[self.last_node().id][self.depot.id]
        self.route.append(self.depot)

    def __str__(self):
        return f'Route: Total cost = {"%.2f" %self.cost}, load = {"%.2f" %self.load}, Customers = \n{self.route}'

    def __repr__(self):
        return str(self)

'''
问题实例
'''
class Instance:
    """
    :param file_name: File bai toan de giai

    Ham nay de dung cac thuoc tinh cho bai toan
        - optimal (float): Gia tri toi uu biet truoc
        - vehicles (int): So xe
        - capacity (int): Tai trong cua xe
        - dimensions (int): So luong khach hang (= so luong yeu cau)
        - customers (list Node): Thong tin cua cac khach hang
        - depot (Node): Diem lay hang cua moi xe
        - distance_matrix (2D float): Ma tran khoang cach giua cac khach hang va kho voi nhau
    """
    def __init__(self, file_name):
        self.optimal = None
        self.vehicles = None
        self.capacity = None
        self.dimensions = 0
        # Customers are mapped with nodes
        self.customers = []
        self.depot = None
        self.distance_matrix = []
        self.load_data(file_name)
        self.create_distance_matrix()

    def load_data(self, file_name):
        print(Config.PROJECT_PATH)
        try:
            f = None
            try:
                f = open(Config.PROJECT_PATH + "/problems/CVRP/" + file_name + ".vrp")
            except FileNotFoundError:
                print("文件不存在")
                exit(-1)

            lines = f.readlines()
            line_two = re.findall(r'\d+',lines[1])
            # Input binh thuong co the khong co thong tin ve optimal
            try:
                self.optimal = int(line_two[1])
            except Exception:
                self.optimal = -1
            self.vehicles = int(line_two[0])
            self.dimensions = int(re.findall(r'\d+', lines[3])[0])
            self.capacity = int(re.findall(r'\d+', lines[5])[0])

            # Tu dong 8 lay thong tin vi tri kho, khach hang va nhu cau cua khach
            for i in range(self.dimensions):
                demand = int(lines[8 + self.dimensions + i].split().pop())
                node = lines[7 + i].split()
                y = node.pop()
                x = node.pop()
                # Truong hop kho. Gia su rang kho se co demand = 0
                if demand == 0:
                    self.depot = Node(i, x, y, demand)
                else:
                    self.customers.append(Node(i, x, y, demand))
        except Exception:
            print("Loi nhap du lieu: Du lieu khong khop dinh dang")
            exit(-1)

    # Tao ma tran khoang cach giua khach hang va depot voi nhau
    def create_distance_matrix(self):
        nodes = [self.depot] + self.customers
        for i in range(len(nodes)):
            row = []
            for j in range(len(nodes)):
                if i == j:
                    row.append(0)
                else:
                    vector = np.array([nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y])
                    row.append(np.linalg.norm(vector))
            self.distance_matrix.append(row)
        self.distance_matrix = np.array(self.distance_matrix)

"""
最近邻
"""
class NearestNeighbor:
    """
    :param instance: Representation cua bai toan

    Ham dung nay dung cac thuoc tinh roi sinh nghiem ban dau, chi tiet cac thuoc tinh:
            - customers (list): luu tru cac Node la khach hang
            - distanceMatrix (2D float): Ma tran khoang cach luu thong tin khoang cach giua 2 khach hang bat ki
            - capacity (int): tai trong cua moi xe
            - numberOfVehicle (int): So luong xe
            - depot (Node): depot trong mo hinh CVRP
            - vehicles (list): Cac tuyen duong danh cho moi xe
    """
    def __init__(self, instance):
        self.instance = instance
        self.customers = instance.customers
        self.distanceMatrix = instance.distance_matrix
        self.numberOfVehicle = instance.vehicles

    def get_initial_solution(self):
        """
        :return Nghiem ko can nhat thiet phai tot, mien la dung nghiem nhanh la duoc

        Ham nay dung nghiem ban dau cho bai toan bang Nearest Neighbor Heuristic
        """
        # solution chua tap hop cac route de tra ve
        solution = []
        current_route = Route(self.instance)

        # lap lai cho den khi toan bo khach hang da duoc phuc vu
        while len(self.customers) != 0:
            # Lay ra khach hang cuoi cung trong route hien tai
            current_customer = current_route.last_node()
            min_distance = 999999999
            closet_node = None

            # Tim ra khach hang co vi tri gan nhat voi khach hang cuoi cung cua route hien tai
            for i in self.customers:
                if current_route.can_insert(i):
                    distance = self.distanceMatrix[current_customer.id][i.id]
                    if distance < min_distance:
                        min_distance = distance
                        closet_node = i

            # Xet khach hang gan nhat. Neu ko null thi them vao route va xoa trong danh sach customer can them vao
            if closet_node is not None:
                current_route.insert(closet_node)
                self.customers.remove(closet_node)
            # Neu khong insert duoc thi sang xe khac de insert. Neu day la xe cuoi cung thi return
            else:
                current_route.finish_route()
                solution.append(current_route)
                current_route = Route(self.instance)

                if len(solution) == self.numberOfVehicle:
                    return Solution(solution)

        # Insert route cuoi cung khi het khach hang
        current_route.finish_route()
        solution.append(current_route)

        # khoi tao not cho cac xe neu chua co khach hang nao
        while len(solution) < self.numberOfVehicle:
            solution.append(Route(self.instance).finish_route())

        return Solution(solution, self.instance)

'''
破坏和修复策略
'''
def get_in_plan(routes, to_remove):
    i = randint(0, len(routes) - 1)
    counter = 1
    while len(routes[i].route) < to_remove + 2:
        try:
            i = randint(0, len(routes) - 1)
            counter += 1
        except counter == 100:
            print("ShawDestroy: Invalid number to removes")
    return i
def rank_using_relatedness(v, visit_sets, distance_matrix):
    visit_list = []
    for visit in visit_sets:
        # Khong xet depot
        if visit.id != 0:
            relate = distance_matrix[v.id][visit.id]
            visit_list.append((visit, relate))
    visit_list.sort(key=lambda x: x[1], reverse=False)
    return visit_list
def shaw_destroy(solution, to_remove, d):
    """
    :param solution: (Solution) the solution to be destroyed
    :param to_remove: (int) the numbers of customers will be removed
    :param d: (float) Deterministic parameter
    :return: partial solution

    Ham nay duoc implement tu bai bao Shaw P. Using constraint programming and local search
    methods to vehicle routing problem. Lecture Notes in Computer Science 1998;1520:417–30
    """
    selected_route = get_in_plan(solution.routes, to_remove)
    visit_sets = deepcopy(solution.routes[selected_route].route)
    v = visit_sets[randint(1, len(visit_sets) - 2)]
    visit_sets.remove(v)
    removed = [v]

    while len(removed) < to_remove:
        v = choice(removed)
        # Rank visits in plan with respect to relatedness to v. Rank will be decreasing order
        lst = rank_using_relatedness(v, visit_sets, solution.instance.distance_matrix)
        rand = random()
        v = lst[int((len(lst) - 1) * rand ** d)][0]
        removed.append(v)
        visit_sets.remove(v)

    for node in removed:
        solution.routes[selected_route].erase(node.id)
        solution.extracted_node.append(node)
    return solution
def random_destroy(solution, to_remove, d=None):
    """
    :param solution: (Solution) the solution to be destroyed
    :param to_remove: (int) the numbers of customers will be removed
    :return: partial solution

    Ham nay duoc xoa ngau nhien customer tu 1 route bat ki
    """
    removed = 0
    while removed < to_remove:
        random_route = choice(solution.routes)
        # Neu route co customer ngoai depot ra
        if len(random_route.route) > 2:
            random_customer = random_route.route[randint(1, len(random_route.route) - 2)]
            random_route.erase(random_customer.id)
            solution.extracted_node.append(random_customer)
            removed += 1

    return solution
def find_worst_position(solution):
    max_cost = 0
    distance_matrix = solution.instance.distance_matrix
    position = None

    for i in range(len(solution.routes)):
        # ko xet 2 cai depot tren 1 route
        for j in range(1, len(solution.routes[i].route) - 2):
            current_node = solution.routes[i].route[j].id
            before_node = solution.routes[i].route[j - 1].id
            after_node = solution.routes[i].route[j + 1].id
            cost = distance_matrix[before_node][current_node] + distance_matrix[current_node][after_node] - distance_matrix[before_node][after_node]
            if cost > max_cost:
                position = (i, j)

    return position
def worst_destroy(solution, to_remove, d=None):
    """
    :param solution: Nghiem can duoc pha
    :param to_remove: So luong khach hang se loai bo
    :return: Nghiem sau khi pha

    Ham nay se loai bo nhung khach hang nao co cost cao nhat
    """
    removed = []
    while len(removed) < to_remove:
        position = find_worst_position(solution)
        remove_node = solution.routes[position[0]].route[position[1]]
        removed.append(remove_node)
        solution.routes[position[0]].erase(remove_node.id)

    solution.extracted_node = removed
    return solution
def massive_destroy(solution, to_remove=None, d=None):
    """
    :param solution: Nghiem can duoc pha
    :return: parital solution

    Ham nay thuc hien xoa tat ca khach hang tu 2 route ngau nhien
    """
    route1 = choice(solution.routes)
    solution.routes.remove(route1)
    route2 = choice(solution.routes)
    solution.routes.remove(route2)

    customers = route1.route + route2.route
    for customer in customers:
        if customer.id != 0:
            solution.extracted_node.append(customer)

    # Bo sung 2 route moi sau khi da xoa 2 route cu di
    a = Route(solution.instance)
    a.finish_route()
    b = Route(solution.instance)
    b.finish_route()
    solution.routes.append(a)
    solution.routes.append(b)

    return solution

def find_best_position(node, solution):
    min_insert_cost = 999999999
    distance_matrix = solution.instance.distance_matrix
    position = None

    for i in range(len(solution.routes)):
        if solution.routes[i].can_insert(node):
            for j in range(0, len(solution.routes[i].route) - 1):
                before_node = solution.routes[i].route[j].id
                after_node = solution.routes[i].route[j + 1].id
                insert_cost = distance_matrix[before_node][node.id] + distance_matrix[node.id][after_node] - \
                              distance_matrix[before_node][after_node]
                if insert_cost < min_insert_cost:
                    position = (i, j)

    return position
def greedy_repair(solution):
    """
    :param solution: (Solution) the solution to be repaired
    :return: None

    Ham nay sua nghiem sau khi da pha
    """
    # Nghiem co mot ti le thap khong sua duoc nen se phai xu ly neu phat sinh
    try:
        for node in solution.extracted_node:
            position = find_best_position(node, solution)
            solution.routes[position[0]].insert(node, position[1])

        solution.extracted_node = []
        solution.update_total_cost()
        return solution
    except Exception:
        pass
def random_repair(solution):
    for node in solution.extracted_node:
        insert_route = []
        for route in solution.routes:
            if route.can_insert(node):
                insert_route.append(route)

        # Nghiem co mot ti le thap khong sua duoc nen se phai xu ly neu phat sinh
        if len(insert_route) == 0:
            return
        else:
            route = choice(insert_route)
            position = randint(0, len(route.route) - 2)
            route.insert(node, position)

    solution.extracted_node = []
    solution.update_total_cost()
    return solution

'''
自适应机制
'''
class AdaptiveMechanism:
    def __init__(self):
        self.destroy_operator = [shaw_destroy, random_destroy, worst_destroy, massive_destroy]
        self.repair_operator = [greedy_repair, random_repair]

        self.destroy_weight = [1, 1, 1, 1]
        self.repair_weight = [1, 1]

        self.destroy_score = [0, 0, 0, 0]
        self.repair_score = [0, 0]

        self.destroy_used = [0, 0, 0, 0]
        self.repair_used = [0, 0]

    def select_operator(self):
        """
        :return: Tra ve 1 cap destroy - repair operator theo co che roulette-wheel (uu tien chon operator co trong so cao)
        """
        return choices(self.destroy_operator, weights=self.destroy_weight, k=1)[0], choices(self.repair_operator, weights=self.repair_weight, k=1)[0]

    def update_score(self, operator, phi):
        """
        :param operator: 1 cap destroy - repair
        :param phi: performance
        :return:

        Ham update diem dua tren performance phi va so lan da su dung operator
        """
        destroy_index = self.destroy_operator.index(operator[0])
        repair_index = self.repair_operator.index(operator[1])
        self.destroy_score[destroy_index] += phi
        self.destroy_used[destroy_index] += 1
        self.repair_score[repair_index] += phi
        self.repair_used[repair_index] += 1

    def update_weight(self):
        """
        :return:

        Ham update trong so cua cac operator dua tren past performance
        """
        # Update destroy
        for i in range(len(self.destroy_operator)):
            if self.destroy_used[i] != 0:
                w = self.destroy_weight[i]
                pi = self.destroy_score[i]
                o = self.destroy_used[i]
                self.destroy_weight[i] = (1 - Config.N) * w + Config.N * pi / (Config.V * o)

        # Update repair
        for i in range(len(self.repair_operator)):
            if self.repair_used[i] != 0:
                w = self.repair_weight[i]
                pi = self.repair_score[i]
                o = self.repair_used[i]
                self.repair_weight[i] = (1 - Config.N) * w + Config.N * pi / (Config.V * o)

        # Reset het diem va so lan da dung
        self.destroy_score = [0, 0, 0, 0]
        self.repair_score = [0, 0]
        self.destroy_used = [0, 0, 0, 0]
        self.repair_used = [0, 0]

"""
局部搜索
"""
def improve(solution):
    cost = solution.instance.distance_matrix
    for route in solution.routes:
        customers = route.route
        if len(customers) >= 4:
            # Toi uu 1 tuyen xe bang cach xem xet viec dao cac duong cho nhau
            for x in range(1, len(customers) - 3):
                for v in range(x + 1, len(customers) - 2):
                    # Danh gia chi phi truoc va sau khi dao 2 duong
                    delta = cost[customers[x - 1].id][customers[v].id] + cost[customers[x].id][customers[v + 1].id] - cost[customers[x - 1].id][customers[x].id] - cost[customers[v].id][customers[v + 1].id]
                    if delta < -0.01:
                        # Trong truong hop chi phi co giam se doi 2 duong (not 2 khach hang)
                        i = x
                        j = v
                        route.cost += delta
                        while i < j:
                            tmp = customers[i]
                            customers[i] = customers[j]
                            customers[j] = tmp
                            i += 1
                            j -= 1
                        return True
    return False
def local_search(solution):
    """

    :param solution:
    :return: solution duoc cai tien boi local search

    Thuat toan local search (Hill climbing) su dung move 2-opt first improve
    """
    improved = True
    while improved:
        improved = improve(solution)
    return solution

class ALNS:
    def __init__(self, filename):
        self.filename = filename


    def run(self):
        adt = AdaptiveMechanism()
        best = NearestNeighbor(Instance(self.filename)).get_initial_solution()
        ips = Config.ITERATION / Config.SEGMENT

        s = deepcopy(best)
        test = 0
        print("ALNS Process...")

        for i in range(Config.ITERATION):
            # test la co che thu nghiem
            test += 1
            s2 = deepcopy(s)
            # Chon ra cap operator tu co che Roulette wheel
            operators = adt.select_operator()
            # destroy
            s2 = operators[0](s2, int(Config.DESTROY_RATIO * s2.instance.dimensions), d=Config.D)
            # repair
            s2 = operators[1](s2)

            # Trong truong hop nghiem s2 sau khi sua khong cp van de gi
            if s2 is not None:
                # Neu nghiem moi tot hon nghiem cu -> accept
                if s2.totalCost < s.totalCost:
                    s2 = local_search(s2)
                    s = deepcopy(s2)
                    # Neu nghiem cu tot hon nghiem tot nhat tim duoc -> cap nhat trang thai ALNS len console
                    if s2.totalCost < best.totalCost:
                        # print(s2.totalCost)
                        test = 0
                        best = deepcopy(s2)
                        # Cap nhat diem dua tren performance
                        adt.update_score(operators, Config.PHI_1)
                    else:
                        adt.update_score(operators, Config.PHI_2)
                else:
                    adt.update_score(operators, Config.PHI_3)

                # Co che chap nhan nghiem dang duoc thu nghiem (Inspire tu thuat toan Record-to-Record)
                if s2.totalCost < best.totalCost + Config.DEVIATION:
                    s = deepcopy(s2)

                # Considering very good effect (+150 for 2k total cost) (+100 for n39k6). Con time thi phat trien
                if test >= 1000 and s2.totalCost <= best.totalCost + Config.BIG_DEVIATION:
                    s = deepcopy(s2)
                    test = 0
            else:
                adt.update_score(operators, Config.PHI_3)

            # Vao dau moi segment, cap nhat trong so cac operator va tien hanh Local Search len nghiem hien tai
            if (i + 1) % ips == 0:
                adt.update_weight()
                s = local_search(s)
                if s.totalCost < best.totalCost:
                    best = deepcopy(s)

        # In ra nghiem tot nhat tim duoc
        print(best)

        # Luu ket qua vao file
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
        if not os.path.exists(Config.PROJECT_PATH + "/cvrp_alns_results/" + self.filename.split(".vrp")[0]):
            os.makedirs(Config.PROJECT_PATH + "/cvrp_alns_results/" + self.filename.split(".vrp")[0])
        f = open(Config.PROJECT_PATH + "/cvrp_alns_results/" + self.filename.split(".vrp")[0] + "/" + current_time + ".txt", "wt")
        f.write(best.__str__())
        f.close()

        for route in best.routes:
            x = []
            y = []
            for node in route.route:
                x.append(node.x)
                y.append(node.y)
            plt.plot(x, y)
            plt.plot(x, y, 'or')
            plt.plot(best.routes[0].route[0].x, best.routes[0].route[0].y, "sk")
        plt.title("VRP Solution (Cost = " + str("%.2f" % best.totalCost) + ")")
        plt.savefig(Config.PROJECT_PATH + "/cvrp_alns_results/" + self.filename.split(".vrp")[0] + "/" + current_time + ".png")
        plt.show()


if __name__ == "__main__":
    alns = ALNS('A-n32-k5')
    alns.run()
    