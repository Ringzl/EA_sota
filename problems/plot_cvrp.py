import networkx as nx
import matplotlib.pyplot as plt


def plot_path(edges, pos_dct):
    # 每个vehicle的路径
    k_path_dct = {}
    for arc in edges:
        k_path_dct.setdefault(arc[2], []).append((arc[0], arc[1]))

    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for k in k_path_dct:
        G.add_edges_from(k_path_dct[k])
    
    # 绘制图中的所有节点  
    nx.draw_networkx_nodes(G, pos=pos_dct, node_color='lightblue', node_size=100)  
    
    # 每条边分配颜色
    colors = [
        'blue', 'orange', 'green', 'red', 
        'purple', 'brown', 'pink', 'gray', 
        'olive', 'cyan'
    ]
    
    # 为每条路径绘制边，使用不同的颜色 
    for k in k_path_dct:
        nx.draw_networkx_edges(G, pos=pos_dct, edgelist=k_path_dct[k], edge_color=colors[k-1], width=2)  
    
    # 绘制标签（可选）  
    nx.draw_networkx_labels(G, pos=pos_dct)  
    
    # 显示图形  
    plt.axis('off')  
    plt.show()