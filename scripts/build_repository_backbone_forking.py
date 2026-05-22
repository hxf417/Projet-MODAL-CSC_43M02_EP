import json
import networkx as nx

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union > 0 :
        return inter/union
    else :
        return 0.0

def build_fork_network(json_filepath, output_filepath, min_similarity=0.05):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
        
    G_forking = nx.Graph()
    forker_sets = {}

    for repo in repo_data:
        repo_name = repo.get("nameWithOwner")
        
        G_forking.add_node(
            repo_name,
            stars=repo.get("stargazerCount", 0),
            language=repo.get("primaryLanguage", "Unknown"),
            forks=repo.get("forkCount", 0)
        )

        forker_sets[repo_name] = set(repo.get("forker_owners", []))

    repo_names = list(forker_sets.keys())
    
    for i in range(len(repo_names)):
        for j in range(i + 1, len(repo_names)):
            jaccard_score = jaccard(forker_sets[repo_names[i]], forker_sets[repo_names[j]])
        
            if jaccard_score >= min_similarity:
                G_forking.add_edge(repo_names[i], repo_names[j], weight=jaccard_score)

    isolated_nodes = list(nx.isolates(G_forking))
    G_forking.remove_nodes_from(isolated_nodes)
    nodes_to_remove = []
    for comp in nx.connected_components(G_forking):
        if len(comp) <= 3:
            nodes_to_remove += list(comp)
    G_forking.remove_nodes_from(nodes_to_remove)

    number_nodes = G_forking.number_of_nodes()
    number_edges = G_forking.number_of_edges()
    mean_degree = 2*number_edges/number_nodes

    print(f"Graph built with nodes: {number_nodes}, edges: {number_edges}, average degree: {mean_degree}")
    nx.write_gexf(G_forking, output_filepath)

if __name__ == "__main__":
    build_fork_network("../data/raw/repo_raw_data_fork.json", "../outputs/graphs/repo/clean_fork_network.gexf", min_similarity=0.025)