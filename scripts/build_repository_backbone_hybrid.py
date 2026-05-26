import json
import re
import networkx as nx
from tqdm import tqdm
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# --- Pre-processing elements from build_repository_backbone_readme_v2.py ---
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

# --- Jaccard similarity from build_repository_backbone_forking.py ---
def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union > 0 :
        return inter/union
    else :
        return 0.0

def dummy_tokenizer(doc):
    return doc

# --- Main Hybrid Network Building Logic ---
def build_hybrid_network(json_filepath, output_filepath, alpha=0.5, beta=0.5, min_similarity=0.05):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
        
    G = nx.Graph()
    readme_tokens_list = []
    repo_names = []
    forker_sets = {}
    
    # 1. Add Nodes & Extract text and forkers
    for repo in tqdm(repo_data, desc="Adding Nodes & Extracting Data"):
        repo_name = repo.get("nameWithOwner")
        repo_readme = repo.get("readme_text", "")
        domains_list = repo.get("source_domains") or []
        
        G.add_node(
            repo_name,
            stars=repo.get("stargazerCount", 0),
            language=repo.get("primaryLanguage", "Unknown"),
            forks=repo.get("forkCount", 0),
            source_domain=", ".join(domains_list)
        )
        
        repo_names.append(repo_name)
        readme_tokens_list.append(preprocess_readme(repo_readme))
        forker_sets[repo_name] = set(repo.get("forker_owners", []))

    N = len(repo_names)

    # 2. Compute Sparse TF-IDF matrices using scikit-learn
    print("Computing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_tokenizer,
        preprocessor=dummy_tokenizer,
        token_pattern=None,
        max_df=0.65,           
        sublinear_tf=True,     
        smooth_idf=True,       
        norm='l2'              
    )
    
    tfidf_matrix = vectorizer.fit_transform(readme_tokens_list)

    # 3. Compute pairwise Cosine Similarities
    print("Computing TF-IDF similarity matrix...")
    sparse_sim_matrix = tfidf_matrix * tfidf_matrix.T
    
    # Extract only the upper triangle (k=1) as a dictionary of keys (DOK) for O(1) lookup
    upper_tri = sp.triu(sparse_sim_matrix, k=1).todok()

    print(f"Computing Hybrid Edges (alpha={alpha}, beta={beta})...")
    # 4. Compute hybrid similarities
    for i in tqdm(range(len(repo_names)), desc="Building Edges"):
        for j in range(i + 1, len(repo_names)):
            fork_sim = jaccard(forker_sets[repo_names[i]], forker_sets[repo_names[j]])
            # Fetch TF-IDF similarity from the sparse matrix (defaults to 0.0 if not present)
            readme_sim = upper_tri.get((i, j), 0.0)
            
            hybrid_sim = alpha * fork_sim + beta * readme_sim
            
            if hybrid_sim >= min_similarity:
                G.add_edge(repo_names[i], repo_names[j], weight=hybrid_sim)

    # 5. Cleanup
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    nodes_to_remove = []
    for comp in nx.connected_components(G):
        if len(comp) <= 3:
            nodes_to_remove += list(comp)
    G.remove_nodes_from(nodes_to_remove)

    number_nodes = G.number_of_nodes()
    number_edges = G.number_of_edges()
    mean_degree = (2 * number_edges / number_nodes) if number_nodes > 0 else 0.0

    print(f"Graph built with nodes: {number_nodes}, edges: {number_edges}, average degree: {mean_degree:.2f}")
    
    nx.write_gexf(G, output_filepath)

if __name__ == "__main__":
    build_hybrid_network(
        "data/raw/repo_raw_data_fork.json", 
        "outputs/graphs/repo/clean_hybrid_network.gexf", 
        alpha=0.5,
        beta=0.5,
        min_similarity=0.05
    )
