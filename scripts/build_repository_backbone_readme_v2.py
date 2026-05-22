import json
import math
import re
import networkx as nx
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Tuple

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

def build_tfidf_vectors(
    docs_tokens: List[List[str]],
    max_df_ratio: float = 0.65,
) -> Tuple[List[Dict[str, float]], List[float]]:
    n_docs = len(docs_tokens)
    df = Counter()
    for tokens in docs_tokens:
        for tok in set(tokens):
            df[tok] += 1

    vectors: List[Dict[str, float]] = []
    norms: List[float] = []
    max_df = max(1, int(math.ceil(n_docs * max_df_ratio)))

    for tokens in docs_tokens:
        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        for tok, count in tf.items():
            if df[tok] > max_df:
                continue
            idf = math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0
            weight = (1.0 + math.log(count)) * idf
            vec[tok] = weight
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors.append(vec)
        norms.append(norm)
    return vectors, norms

def cosine_similarity_sparse(
    vec_a: Dict[str, float],
    norm_a: float,
    vec_b: Dict[str, float],
    norm_b: float,
) -> float:
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    # Ensure vec_a is the smaller dictionary to iterate faster
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a
    dot = 0.0
    for token, weight_a in vec_a.items():
        weight_b = vec_b.get(token)
        if weight_b:
            dot += weight_a * weight_b
    score = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, score))

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

    # 2. Compute Sparse TF-IDF matrices
    print("Computing TF-IDF vectors...")
    tfidf_vectors, tfidf_norms = build_tfidf_vectors(readme_tokens_list)

    # 3. Compute pairwise Cosine Similarities & Add Edges
    total_pairs = N * (N - 1) // 2
    for i, j in tqdm(combinations(range(N), 2), desc="Adding Edges", total=total_pairs):
        readme_similarity_score = cosine_similarity_sparse(
            tfidf_vectors[i], tfidf_norms[i], 
            tfidf_vectors[j], tfidf_norms[j]
        )
    
        if readme_similarity_score >= min_similarity:
            G.add_edge(repo_names[i], repo_names[j], weight=readme_similarity_score)

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
        "../data/raw/repo_raw_data_fork.json", 
        "../outputs/graphs/repo/clean_readme_network.gexf", 
        min_similarity=0.1 
    )