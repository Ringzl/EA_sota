import networkx as nx
import matplotlib.pyplot as plt
import time

def plot_path(edges, pos_dct):
    # Create a graph
    G = nx.Graph()
    # Add edges to the graph
    G.add_edges_from(edges)

    # Calculate the positions of the nodes using a spring layout
    # pos = nx.spring_layout(G)

    # Define visualization options
    options = {
        "font_size": 5,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "black",
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
    time.sleep(3)
    plt.close()
