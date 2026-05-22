import json
import networkx as nx
import math
import numpy as np
from tqdm import tqdm
import re
import markdown
from bs4 import BeautifulSoup
import matplotlib as plt

def strip_readme_to_text(md_string):
    # 1. Remove code blocks (fenced blocks with ```)
    text = re.sub(r'```[\s\S]*?```', '', md_string)
    
    # 2. Remove inline code (single backticks)
    text = re.sub(r'`[^`\n]+`', '', text)
    
    # 3. Remove HTML comments (like the <!-- Android... --> in the example)
    text = re.sub(r'<!--[\s\S]*?-->', '', text)

    # 4. Convert the remaining Markdown to HTML
    html = markdown.markdown(text, extensions=['tables'])
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text(separator=' ')
    lines = [line.strip() for line in plain_text.splitlines()]
    plain_text = '\n'.join(lines)
    plain_text = re.sub(r'\n{3,}', '\n\n', plain_text)
    return plain_text.strip()

ai_dict = {}
ai_dict_data = {}
with open("../dict_testing.json", "r", encoding = "utf-8") as f:
    ai_dict_data = json.load(f)

for concept in ai_dict_data["results"]["bindings"]:
    ai_dict[concept["conceptLabel"]["value"].casefold()] = 0

def tfidf(readme: str)->np.array:
    v_readme = np.zeros(len(ai_dict))
    for i, word in enumerate(ai_dict):
        v_readme[i] = readme.count(word) * ai_dict[word]
    return v_readme

def cos_similarity(v1: np.array, v2: np.array)->float:
    if np.linalg.norm(v1)*np.linalg.norm(v2) == 0:
        return 0.0
    return np.sum(v1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def build_readme_network(json_filepath, output_filepath, min_similarity=0.05):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
        
    G = nx.Graph()
    readme_corpus = {}
    N = len(repo_data)
    for repo in tqdm(repo_data, desc="Adding the Nodes"):
        repo_name = repo.get("nameWithOwner")
        repo_readme = repo.get("readme_text")
        
        G.add_node(
            repo_name,
            stars=repo.get("stargazerCount", 0),
            language=repo.get("primaryLanguage", "Unknown"),
            forks=repo.get("forkCount", 0)
        )
        
        readme_corpus[repo_name] = strip_readme_to_text(repo_readme).casefold()

    for word in tqdm(ai_dict.keys(), desc="Computing IDF"):
        for readme in readme_corpus.values():
            if word in readme:
                ai_dict[word] += 1
        if ai_dict[word] > 0 :
            ai_dict[word] = math.log10(N/ai_dict[word])

    repo_names = list(readme_corpus.keys())

    tfidf_matrix = np.zeros((len(ai_dict), N))
    for j in tqdm(range(N), desc="Computing TF-IDF matrix"):
        tfidf_matrix[ : , j] = tfidf(readme_corpus[repo_names[j]])

    for i in tqdm(range(N), desc = "Adding the Edges"):
        for j in range(i + 1, N):
            readme_similarity_score = cos_similarity(tfidf_matrix[:, i], tfidf_matrix[:, j])
        
            if readme_similarity_score >= min_similarity:
                G.add_edge(repo_names[i], repo_names[j], weight=readme_similarity_score)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # nodes_to_remove = []
    # for comp in nx.connected_components(G):
    #     if len(comp) <= 3:
    #         nodes_to_remove += list(comp)
    # G.remove_nodes_from(nodes_to_remove)

    number_nodes = G.number_of_nodes()
    number_edges = G.number_of_edges()
    mean_degree = 2*number_edges/number_nodes

    print(f"Graph built with nodes: {number_nodes}, edges: {number_edges}, average degree: {mean_degree}")
    nx.write_gexf(G, output_filepath)

if __name__ == "__main__":
    
    build_readme_network("../data/raw/repo_raw_data_fork.json", "../outputs/graphs/repo/clean_readme_network.gexf", min_similarity=0.9)