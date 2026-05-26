import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def basic_metric_analyser(graph_filepath):
    graph = nx.read_gexf(graph_filepath)

    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)
    mean_degree = 2*n_edges/n_nodes
    density = n_edges/(n_nodes*(n_nodes - 1))

    n_connected_components = nx.number_connected_components(graph)

    print(f"""BASIC METRIC ANALYSIS
                - number of nodes : {n_nodes}
                - number of edges : {n_edges}
                - mean_degree : {mean_degree}
                - density : {density}
                - number of CC : {n_connected_components}""")
    

def centrality_analyser(graph_filepath):
    graph = nx.read_gexf(graph_filepath)

    betweeness_dict = nx.betweenness_centrality(graph)
    eigenvector_dict = nx.eigenvector_centrality(graph)
    degree_dict = nx.degree_centrality(graph)

    betweeness_top_10 = Counter(betweeness_dict).most_common(10)
    eigenvector_top_10 = Counter(eigenvector_dict).most_common(10)
    degree_top_10 = Counter(degree_dict).most_common(10)

    print("\n\n\nBETWEENESS CENTRALITY")
    for node, val in betweeness_top_10:
        print(f"{node} ({val})")

    print("\n\n\nEIGENVECTOR CENTRALITY")
    for node, val in eigenvector_top_10:
        print(f"{node} ({val})")

    print("\n\n\nDEGREE CENTRALITY")
    for node, val in degree_top_10:
        print(f"{node} ({val})")
    

if __name__ == "__main__":
    
    basic_metric_analyser("outputs/graphs/repo/clean_fork_network.gexf")
    centrality_analyser("outputs/graphs/repo/clean_fork_network.gexf")