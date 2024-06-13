import re

class CVRP:
    def __init__(self, path):
        capacity, graph, demands, optimalValue, name, n_trucks, n_nodes = self.load(path)
        self.name = name
        self.n_trucks = n_trucks
        self.n_nodes = n_nodes
        self.capacity = capacity
        self.demands = demands
        self.graph = graph
        self.optimalValue = optimalValue
    
    def load(self, path):
        f = open(path, "r")
        content = f.read()
        name = content.split()[2].rstrip()
        n_trucks = re.search(r"No of trucks: (\d+)", content, re.MULTILINE)
        n_trucks = n_trucks.group(1)
        optimalValue = re.search(r"Optimal value: (\d+)", content, re.MULTILINE)
        if(optimalValue != None):
            optimalValue = optimalValue.group(1)
        else:
            optimalValue = re.search(r"Best value: (\d+)", content, re.MULTILINE)
            if(optimalValue != None):
                optimalValue = optimalValue.group(1)
        capacity = re.search(r"^CAPACITY : (\d+)$", content, re.MULTILINE).group(1)
        graph = re.findall(r"^ (\d+) (\d+) (\d+)$", content, re.MULTILINE)
        demand = re.findall(r"^(\d+) (\d+) $", content, re.MULTILINE)
        graph = {int(a):(int(b),int(c)) for a,b,c in graph}
        demand = {int(a):int(b) for a,b in demand}
        capacity = int(capacity)
        optimalValue = int(optimalValue)
        n_trucks = int(n_trucks)
        n_nodes = len(graph)

        return capacity, graph, demand, optimalValue, name, n_trucks, n_nodes
        


if __name__ == "__main__":
    fpath = "/home/yongcun/work/optimize/ec/problems/CVRP/A-n32-k5.vrp"
    prob = CVRP(fpath)
    print(prob.demands)
    print('end')

