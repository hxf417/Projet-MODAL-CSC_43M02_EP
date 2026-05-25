import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def threshold_analyser(graph_filepath, min_value, max_value):
    graph = nx.read_gexf(graph_filepath)
    threshold_values = np.linspace(min_value, max_value, num = 100)
    nodes_values = np.zeros(100)
    edges_values = np.zeros(100)
    for i, eps in enumerate(threshold_values):
        for u, v, weight in list(graph.edges(data = "weight")):
            if weight < eps:
                graph.remove_edge(u, v)
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)
        nodes_values[i] = graph.number_of_nodes()
        edges_values[i] = graph.number_of_edges()
    mean_degree_values = 2*edges_values/nodes_values

    # Plotting the graph evolution
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)    
    ax1.plot(threshold_values, nodes_values, label="Number of nodes")
    ax1.plot(threshold_values, edges_values, label="Number of edges")
    ax2.plot(threshold_values, mean_degree_values, label="Mean degree")

    ax1.set_title("Number of nodes and edges vs edge threshold")
    ax1.set_xlabel("threshold value")
    ax1.legend()

    ax2.set_title("Graph density vs edge threshold")
    ax2.set_xlabel("threshold value")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    
    threshold_analyser("outputs/graphs/repo/clean_fork_network.gexf", 0.01, 0.03)