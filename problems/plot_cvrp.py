import networkx as nx
import matplotlib.pyplot as plt


def plot_path(edges, pos_dct):
    # 每个vehicle的路径
    k_path_dct = {}
    for arc in edges:
        k_path_dct.setdefault(arc[0], []).append((arc[1], arc[2]))

    # 每条边分配颜色
    colors = [
        'blue', 'orange', 'green', 'red', 
        'purple', 'brown', 'pink', 'gray', 
        'olive', 'cyan'
    ]
    edge_colors = []

    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for k in k_path_dct:
        G.add_edges_from(k_path_dct[k])
        edge_colors += [colors[k-1] for _ in range(len(k_path_dct[k]))]

    # Calculate the positions of the nodes using a spring layout
    # pos = nx.spring_layout(G)


    # Define visualization options
    options = {
        "font_size": 5,
        "node_size": 100,
        "node_color": "white",
        "edge_color": edge_colors,
        "linewidths": 1,
        "width": 1,
    }

    # Draw the graph with options
    nx.draw_networkx(G, pos=pos_dct, **options)

    # Adjust margins before disabling axes
    ax = plt.gca()        
    ax.margins(0.20)

    # Turn off axes and show the plot
    plt.axis("off")
    plt.show()