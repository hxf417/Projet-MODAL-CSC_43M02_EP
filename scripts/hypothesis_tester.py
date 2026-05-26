import networkx as nx
import scipy.stats as sps
import pandas as pd

def test_centrality_and_crossdomain(graph_filepath):
    graph = nx.read_gexf(graph_filepath)
    betweeness_dict = nx.betweenness_centrality(graph)
    cross_domain_repo = []
    single_domain_repo = []

    for node, node_data in graph.nodes(data=True):
        domain_string = node_data.get("source_domain", "")
        assert(domain_string)
        domain_list = domain_string.split(", ")
        if len(domain_list) == 2:
            cross_domain_repo.append(betweeness_dict[node])
        else:
            single_domain_repo.append(betweeness_dict[node])
    
    _, pval_1 = sps.shapiro(cross_domain_repo)
    _, pval_2 = sps.shapiro(single_domain_repo)

    norm_distributed = (pval_1 >= 0.05) and (pval_2 >= 0.05)
    conducted_test = "Independent T-test" if norm_distributed else "Mann and Whitney test"

    if norm_distributed:
        result = sps.ttest_ind(cross_domain_repo, single_domain_repo)
    else :
        result = sps.mannwhitneyu(cross_domain_repo, single_domain_repo)

    if result.pvalue < 0.05 :
        print(f"The {conducted_test} has confirmed the two groups differ")
    else:
        print(f"The {conducted_test} has failed to confirm the two groups differ")

def test_central_repo_stars(graph_filepath):
    graph = nx.read_gexf(graph_filepath)
    eigenvector_dict = nx.eigenvector_centrality(graph)

    centrality_list = []
    star_list = []

    for node, node_data in graph.nodes(data=True):
        centrality_list.append(eigenvector_dict[node])
        star_list.append(node_data.get("stars"))
    
    result = sps.pearsonr(centrality_list, star_list)

    if result.pvalue < 0.05:
        if result.statistic > 0.7:
            print("The Pearson test confirmed the hypothesis with strong correlation")
        elif result.statistic > 0.4:
            print("The Pearson test confirmed the hypothesis with moderate correlation")
        elif result.statistic > 0:
            print("The Pearson test confirmed the hypothesis with weak correlation")
        else:
            print("The Pearson test confirmed the hypothesis with negative correlation??")
    else:
        print("The Pearson test failed to confirm the hypothesis of correlation")



if __name__ == "__main__":
    
    test_central_repo_stars("outputs/graphs/repo/clean_fork_network.gexf")
