import json
import math
import re
import networkx as nx
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# --- Pre-processing elements from build_repository_backbone.py ---
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}")
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "your", "you",
    "are", "our", "their", "using", "use", "used", "into", "about",
    "have", "has", "was", "were", "will", "can", "all", "any", "new",
    "more", "most", "also", "not", "but", "via", "http", "https",
    "www", "github", "project", "repository", "code", "model",
    "models", "learning", "machine", "deep", "ai",
}

def preprocess_readme(text: str) -> List[str]:
    if not text:
        return []
    # Strip basic markdown elements
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^\)]*\)", " ", text)
    
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens

# --- Main Network Building Logic ---
def build_readme_network(json_filepath, output_filepath, min_similarity=0.05):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
        
    G = nx.Graph()
    readme_tokens_list = []
    repo_names = []
    
    # 1. Add Nodes & Extract text
    for repo in tqdm(repo_data, desc="Adding Nodes & Preprocessing READMEs"):
        repo_name = repo.get("nameWithOwner")
        repo_readme = repo.get("readme_text", "")
        
        G.add_node(
            repo_name,
            stars=repo.get("stargazerCount", 0),
            language=repo.get("primaryLanguage", "Unknown"),
            forks=repo.get("forkCount", 0)
        )
        
        repo_names.append(repo_name)
        readme_tokens_list.append(preprocess_readme(repo_readme))

    N = len(repo_names)

    # 2. Compute Sparse TF-IDF matrices using scikit-learn
    print("Computing TF-IDF vectors...")
    
    # Trick to let scikit-learn accept your pre-tokenized lists directly
    def dummy_tokenizer(doc):
        return doc

    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_tokenizer,
        preprocessor=dummy_tokenizer,
        token_pattern=None,
        max_df=0.65,           # Matches your max_df_ratio
        sublinear_tf=True,     # Applies the (1 + log(tf)) formula!
        smooth_idf=True,       # Applies the +1 smoothing to IDF
        norm='l2'              # Normalizes vectors so dot product = cosine similarity
    )
    
    # This outputs a highly compressed Scipy sparse matrix
    tfidf_matrix = vectorizer.fit_transform(readme_tokens_list)

    # 3. Compute pairwise Cosine Similarities & Add Edges
    print("Computing similarity matrix...")
    
    # Because vectors are L2-normalized, the dot product IS the cosine similarity.
    # Multiplying the matrix by its transpose calculates all similarities instantly in C.
    sparse_sim_matrix = tfidf_matrix * tfidf_matrix.T

    # Extract only the upper triangle (k=1) to avoid duplicate pairs and self-loops
    upper_tri = sp.triu(sparse_sim_matrix, k=1)
    rows, cols, weights = sp.find(upper_tri)

    print("Adding Edges to Graph...")
    # Add edges directly from the sparse matrix results
    for i, j, weight in zip(rows, cols, weights):
        if weight >= min_similarity:
            G.add_edge(repo_names[i], repo_names[j], weight=weight)

    # 4. Cleanup
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    number_nodes = G.number_of_nodes()
    number_edges = G.number_of_edges()
    mean_degree = (2 * number_edges / number_nodes) if number_nodes > 0 else 0.0

    print(f"Graph built with nodes: {number_nodes}, edges: {number_edges}, average degree: {mean_degree:.2f}")
    
    nx.write_gexf(G, output_filepath)

if __name__ == "__main__":
    # Note: Using the 0.9 default from the original script will likely result in 
    # a very sparse or empty graph with the new method. Consider lowering it to 0.05 - 0.2
    build_readme_network(
        "data/raw/repo_raw_data_fork.json", 
        "outputs/graphs/repo/clean_readme_network.gexf", 
        min_similarity=0.1 
    )