#!/usr/bin/env python3
"""Build an AI technology backbone network from GitHub GraphQL data.

Pipeline:
1) Fetch top-star repositories for topic:computer-vision and topic:nlp.
2) Fetch per-repository details (topics + mentionable users).
3) Build elite-user -> topic bipartite graph, then project to topic-topic.
4) Detect topic communities with NetworkX and export community tables/files.
5) Enrich node attributes and export Gephi-compatible GEXF.
6) Print core network statistics and top bridge technologies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
DOMAIN_CV = "computer-vision"
DOMAIN_NLP = "nlp"
DOMAINS = (DOMAIN_CV, DOMAIN_NLP)
REPO_LIMIT_PER_DOMAIN = 100
REPOS_PAGE_SIZE = 50
MENTIONABLE_USERS_LIMIT = 50
REPO_TOPICS_LIMIT = 100
REQUEST_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30
SEMANTIC_LABEL_RULES = {
    "LLM / NLP": {
        "llm",
        "large-language-models",
        "transformer",
        "transformers",
        "nlp",
        "natural-language-processing",
        "language-model",
        "language-models",
        "llama",
        "prompt-engineering",
        "chatbot",
        "chatbots",
        "conversational-ai",
        "text-generation",
        "instruction-tuning",
    },
    "Computer Vision": {
        "computer-vision",
        "image-processing",
        "image-classification",
        "object-detection",
        "segmentation",
        "instance-segmentation",
        "image-segmentation",
        "vision",
        "opencv",
        "yolo",
        "ocr",
        "classification",
    },
    "Multimodal AI": {
        "multimodal",
        "vlm",
        "vision-language-model",
        "vision-language",
        "text-to-image",
        "image-to-text",
        "diffusion",
        "generative-ai",
        "text-generation-webui",
    },
    "ML Frameworks": {
        "deep-learning",
        "machine-learning",
        "pytorch",
        "tensorflow",
        "keras",
        "jax",
        "neural-network",
        "deep-neural-networks",
        "ai",
        "data-science",
    },
    "Robotics / Autonomous Driving": {
        "autonomous-driving",
        "robotics",
        "ros",
        "carla",
        "carla-simulator",
        "imitation-learning",
        "reinforcement-learning",
        "drone",
    },
    "Information Extraction": {
        "named-entity-recognition",
        "semantic-parsing",
        "dependency-parser",
        "pos-tagging",
        "information-extraction",
        "corpus-builder",
        "corpus-tools",
        "news-crawler",
        "news-aggregator",
        "readability",
        "html-to-markdown",
    },
}


LIST_REPOS_QUERY = """
query($searchQuery: String!, $cursor: String, $first: Int!) {
  search(query: $searchQuery, type: REPOSITORY, first: $first, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        nameWithOwner
        stargazerCount
        forkCount
        primaryLanguage { name }
      }
    }
  }
}
"""


REPO_DETAILS_QUERY = """
query($owner: String!, $name: String!, $topicsFirst: Int!, $usersFirst: Int!) {
  repository(owner: $owner, name: $name) {
    repositoryTopics(first: $topicsFirst) {
      nodes { topic { name } }
    }
    mentionableUsers(first: $usersFirst) {
      nodes { login }
    }
  }
}
"""


def graphql_request(
    query: str,
    variables: Dict,
    headers: Dict[str, str],
    session: requests.Session,
) -> Dict:
    """Send GraphQL request with retry/backoff for transient failures."""
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.post(
                GITHUB_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 500:
                raise requests.HTTPError(
                    f"GitHub server error: HTTP {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(f"GraphQL errors: {payload['errors']}")
            return payload["data"]
        except Exception:
            if attempt == REQUEST_RETRIES:
                raise
            sleep_seconds = RETRY_BACKOFF_SECONDS * attempt
            time.sleep(sleep_seconds)
    raise RuntimeError("Unreachable retry flow.")


def fetch_top_repositories(
    domain: str,
    limit: int,
    headers: Dict[str, str],
    session: requests.Session,
) -> List[Dict]:
    """Fetch top repositories for a domain using paginated search."""
    repos: List[Dict] = []
    cursor: Optional[str] = None
    search_query = f"topic:{domain} sort:stars-desc"

    while len(repos) < limit:
        first = min(REPOS_PAGE_SIZE, limit - len(repos))
        data = graphql_request(
            LIST_REPOS_QUERY,
            {"searchQuery": search_query, "cursor": cursor, "first": first},
            headers,
            session,
        )
        search_data = data["search"]
        nodes = search_data["nodes"] or []
        if not nodes:
            break
        repos.extend(nodes)

        page_info = search_data["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
        time.sleep(0.2)
    return repos[:limit]


def fetch_repository_details(
    repos: Iterable[str],
    headers: Dict[str, str],
    session: requests.Session,
) -> Dict[str, Dict]:
    """Fetch repository topics and mentionable users for each repository."""
    details: Dict[str, Dict] = {}
    repo_names = list(repos)
    total = len(repo_names)

    for idx, name_with_owner in enumerate(repo_names, start=1):
        owner, name = name_with_owner.split("/", 1)
        data = graphql_request(
            REPO_DETAILS_QUERY,
            {
                "owner": owner,
                "name": name,
                "topicsFirst": REPO_TOPICS_LIMIT,
                "usersFirst": MENTIONABLE_USERS_LIMIT,
            },
            headers,
            session,
        )
        details[name_with_owner] = data["repository"] or {}
        if idx % 20 == 0 or idx == total:
            print(f"[detail] fetched {idx}/{total} repositories")
        time.sleep(0.15)
    return details


def clean_users(raw_logins: Iterable[str]) -> List[str]:
    cleaned = []
    for login in raw_logins:
        lower = login.lower()
        if "[bot]" in lower:
            continue
        if lower == "web-flow":
            continue
        cleaned.append(login)
    return cleaned


def build_dataset(token: str) -> List[Dict]:
    """Return merged repository dataset with source domains and details."""
    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()

    repos_by_domain: Dict[str, List[Dict]] = {}
    for domain in DOMAINS:
        print(f"[list] fetching top {REPO_LIMIT_PER_DOMAIN} repos for {domain}")
        repos = fetch_top_repositories(
            domain=domain,
            limit=REPO_LIMIT_PER_DOMAIN,
            headers=headers,
            session=session,
        )
        repos_by_domain[domain] = repos
        print(f"[list] got {len(repos)} repos for {domain}")

    merged_repo_index: Dict[str, Dict] = {}
    for domain, repos in repos_by_domain.items():
        for repo in repos:
            key = repo["nameWithOwner"]
            if key not in merged_repo_index:
                merged_repo_index[key] = {
                    "nameWithOwner": key,
                    "stargazerCount": repo.get("stargazerCount", 0),
                    "forkCount": repo.get("forkCount", 0),
                    "primaryLanguage": (
                        repo.get("primaryLanguage", {}) or {}
                    ).get("name"),
                    "source_domains": set(),
                }
            merged_repo_index[key]["source_domains"].add(domain)

    print(
        f"[merge] requested 200 repos, unique repositories after merge: "
        f"{len(merged_repo_index)}"
    )

    details = fetch_repository_details(
        repos=merged_repo_index.keys(),
        headers=headers,
        session=session,
    )

    final_data = []
    for repo_name, repo_record in merged_repo_index.items():
        detail = details.get(repo_name, {})
        topics = [
            node["topic"]["name"]
            for node in (detail.get("repositoryTopics", {}) or {}).get("nodes", [])
            if node and node.get("topic") and node["topic"].get("name")
        ]
        users = [
            node["login"]
            for node in (detail.get("mentionableUsers", {}) or {}).get("nodes", [])
            if node and node.get("login")
        ]
        final_data.append(
            {
                **repo_record,
                "source_domains": sorted(repo_record["source_domains"]),
                "topics": sorted(set(topics)),
                "mentionableUsers": clean_users(users),
            }
        )
    return final_data


def build_backbone_graph(repo_data: List[Dict]) -> Tuple[nx.Graph, Dict[str, int]]:
    """Build topic-topic backbone graph and return it with elite user stats."""
    user_repo_map: Dict[str, Set[str]] = defaultdict(set)
    topic_stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "total_stars": 0,
            "total_forks": 0,
            "languages": Counter(),
            "domains": set(),
            "repo_count": 0,
        }
    )

    for repo in repo_data:
        repo_name = repo["nameWithOwner"]
        topics = set(repo.get("topics", []))
        users = set(repo.get("mentionableUsers", []))
        stars = int(repo.get("stargazerCount") or 0)
        forks = int(repo.get("forkCount") or 0)
        lang = repo.get("primaryLanguage") or "Unknown"
        domains = set(repo.get("source_domains", []))

        for user in users:
            user_repo_map[user].add(repo_name)

        for topic in topics:
            stats = topic_stats[topic]
            stats["total_stars"] += stars
            stats["total_forks"] += forks
            stats["languages"][lang] += 1
            stats["domains"].update(domains)
            stats["repo_count"] += 1

    elite_users = {u for u, repos in user_repo_map.items() if len(repos) >= 2}
    print(f"[elite] elite users (>=2 repositories): {len(elite_users)}")

    user_topic_map: Dict[str, Set[str]] = defaultdict(set)
    for repo in repo_data:
        topics = set(repo.get("topics", []))
        for user in set(repo.get("mentionableUsers", [])):
            if user in elite_users:
                user_topic_map[user].update(topics)

    bipartite_graph = nx.Graph()
    for user, topics in user_topic_map.items():
        bipartite_graph.add_node(user, bipartite="user")
        for topic in topics:
            bipartite_graph.add_node(topic, bipartite="topic")
            bipartite_graph.add_edge(user, topic)

    topic_nodes = [n for n, d in bipartite_graph.nodes(data=True) if d["bipartite"] == "topic"]
    topic_graph = nx.bipartite.weighted_projected_graph(
        bipartite_graph,
        topic_nodes,
    )

    weak_edges = [(u, v) for u, v, d in topic_graph.edges(data=True) if d.get("weight", 0) < 2]
    topic_graph.remove_edges_from(weak_edges)
    topic_graph.remove_nodes_from(list(nx.isolates(topic_graph)))

    for topic in topic_graph.nodes():
        stats = topic_stats.get(topic, {})
        domains = stats.get("domains", set())
        if DOMAIN_CV in domains and DOMAIN_NLP in domains:
            group = "Cross-domain"
        elif DOMAIN_CV in domains:
            group = "CV"
        elif DOMAIN_NLP in domains:
            group = "NLP"
        else:
            group = "Unknown"

        lang_counter: Counter = stats.get("languages", Counter())
        dominant_language = lang_counter.most_common(1)[0][0] if lang_counter else "Unknown"

        total_stars = int(stats.get("total_stars", 0))
        total_forks = int(stats.get("total_forks", 0))
        repo_count = int(stats.get("repo_count", 0))

        # Gephi-facing attributes with explicit names requested by the user.
        topic_graph.nodes[topic]["Stars"] = total_stars
        topic_graph.nodes[topic]["Group"] = group
        topic_graph.nodes[topic]["Language"] = dominant_language
        topic_graph.nodes[topic]["NodeSize"] = total_stars
        topic_graph.nodes[topic]["Forks"] = total_forks
        topic_graph.nodes[topic]["RepoCount"] = repo_count

        # Lowercase aliases for easier programmatic use.
        topic_graph.nodes[topic]["total_stars"] = total_stars
        topic_graph.nodes[topic]["total_forks"] = total_forks
        topic_graph.nodes[topic]["group"] = group
        topic_graph.nodes[topic]["language"] = dominant_language
        topic_graph.nodes[topic]["repo_count"] = repo_count

    pagerank = nx.pagerank(topic_graph, weight="weight") if topic_graph.number_of_nodes() else {}

    distance_graph = topic_graph.copy()
    for u, v, data in distance_graph.edges(data=True):
        weight = data.get("weight", 1.0)
        data["distance"] = 1.0 / max(float(weight), 1e-9)
    betweenness = (
        nx.betweenness_centrality(distance_graph, weight="distance")
        if topic_graph.number_of_nodes()
        else {}
    )

    partition, community_method = detect_communities(topic_graph)

    for topic in topic_graph.nodes():
        pr_score = float(pagerank.get(topic, 0.0))
        bw_score = float(betweenness.get(topic, 0.0))
        semantic_label = infer_semantic_label(topic)
        topic_graph.nodes[topic]["PageRank"] = pr_score
        topic_graph.nodes[topic]["Betweenness"] = bw_score
        topic_graph.nodes[topic]["pagerank"] = pr_score
        topic_graph.nodes[topic]["betweenness"] = bw_score
        topic_graph.nodes[topic]["community"] = int(partition.get(topic, 0))
        topic_graph.nodes[topic]["community_method"] = community_method
        topic_graph.nodes[topic]["semantic_label"] = semantic_label
        topic_graph.nodes[topic]["SemanticLabel"] = semantic_label

    bridge_nodes = []
    for node in topic_graph.nodes():
        node_comm = partition.get(node, 0)
        has_cross_edge = any(partition.get(nbr, 0) != node_comm for nbr in topic_graph.neighbors(node))
        if has_cross_edge:
            bridge_nodes.append(
                {
                    "topic": node,
                    "betweenness": betweenness.get(node, 0.0),
                    "pagerank": pagerank.get(node, 0.0),
                    "community": node_comm,
                    "group": topic_graph.nodes[node].get("group", "Unknown"),
                }
            )

    bridge_df = pd.DataFrame(bridge_nodes)
    if not bridge_df.empty:
        bridge_df = bridge_df.sort_values(
            by=["betweenness", "pagerank"],
            ascending=[False, False],
        )
        print("\nTop 5 bridge technologies (between communities):")
        print(bridge_df.head(5).to_string(index=False))
    else:
        print("\nTop 5 bridge technologies (between communities): none")

    return topic_graph, {
        "elite_user_count": len(elite_users),
        "community_method": community_method,
    }


def detect_communities(graph: nx.Graph) -> Tuple[Dict[str, int], str]:
    """Detect communities with NetworkX and return node -> community mapping."""
    if graph.number_of_nodes() == 0:
        return {}, "none"

    if graph.number_of_edges() == 0:
        return {node: idx for idx, node in enumerate(graph.nodes())}, "connected-components"

    louvain_communities = getattr(nx.community, "louvain_communities", None)
    if callable(louvain_communities):
        communities = louvain_communities(graph, weight="weight", seed=42)
        method = "networkx-louvain"
    else:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        method = "networkx-greedy-modularity"

    partition: Dict[str, int] = {}
    for idx, community_nodes in enumerate(
        sorted(communities, key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
    ):
        for node in sorted(community_nodes):
            partition[node] = idx
    return partition, method


def build_community_outputs(
    graph: nx.Graph,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Build tabular community outputs from the annotated topic graph."""
    membership_columns = [
        "community_id",
        "topic",
        "pagerank",
        "betweenness",
        "group",
        "semantic_label",
        "language",
        "total_stars",
        "total_forks",
        "repo_count",
        "degree",
    ]
    summary_columns = [
        "community_id",
        "size",
        "internal_edges",
        "density",
        "total_stars",
        "total_forks",
        "total_repo_count",
        "dominant_group",
        "domain_purity",
        "domain_entropy",
        "dominant_semantic_label",
        "proposed_label",
        "semantic_purity",
        "semantic_entropy",
        "top_topics",
    ]
    membership_rows: List[Dict] = []
    community_rows: List[Dict] = []

    community_to_nodes: Dict[int, List[str]] = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        community_to_nodes[int(attrs.get("community", 0))].append(node)

    for community_id, nodes in sorted(community_to_nodes.items()):
        sorted_nodes = sorted(
            nodes,
            key=lambda node: (
                -float(graph.nodes[node].get("pagerank", 0.0)),
                -int(graph.nodes[node].get("total_stars", 0)),
                node,
            ),
        )

        for node in sorted_nodes:
            attrs = graph.nodes[node]
            membership_rows.append(
                {
                    "community_id": community_id,
                    "topic": node,
                    "pagerank": float(attrs.get("pagerank", 0.0)),
                    "betweenness": float(attrs.get("betweenness", 0.0)),
                    "group": attrs.get("group", "Unknown"),
                    "semantic_label": attrs.get("semantic_label", "General AI"),
                    "language": attrs.get("language", "Unknown"),
                    "total_stars": int(attrs.get("total_stars", 0)),
                    "total_forks": int(attrs.get("total_forks", 0)),
                    "repo_count": int(attrs.get("repo_count", 0)),
                    "degree": int(graph.degree(node)),
                }
            )

        community_subgraph = graph.subgraph(sorted_nodes).copy()
        total_stars = sum(int(graph.nodes[node].get("total_stars", 0)) for node in sorted_nodes)
        total_forks = sum(int(graph.nodes[node].get("total_forks", 0)) for node in sorted_nodes)
        total_repo_count = sum(int(graph.nodes[node].get("repo_count", 0)) for node in sorted_nodes)
        top_topics = sorted_nodes[:10]
        domain_counter = Counter(graph.nodes[node].get("group", "Unknown") for node in sorted_nodes)
        dominant_group, domain_purity = compute_purity(domain_counter)
        semantic_scores = score_semantic_labels(sorted_nodes, graph)
        dominant_semantic_label, semantic_purity = compute_purity(semantic_scores)
        proposed_label = choose_proposed_label(dominant_group, dominant_semantic_label)
        community_rows.append(
            {
                "community_id": community_id,
                "size": len(sorted_nodes),
                "internal_edges": community_subgraph.number_of_edges(),
                "density": nx.density(community_subgraph) if len(sorted_nodes) > 1 else 0.0,
                "total_stars": total_stars,
                "total_forks": total_forks,
                "total_repo_count": total_repo_count,
                "dominant_group": dominant_group,
                "domain_purity": domain_purity,
                "domain_entropy": compute_entropy(domain_counter),
                "dominant_semantic_label": dominant_semantic_label,
                "proposed_label": proposed_label,
                "semantic_purity": semantic_purity,
                "semantic_entropy": compute_entropy(semantic_scores),
                "top_topics": ", ".join(top_topics),
            }
        )

    membership_df = pd.DataFrame(membership_rows, columns=membership_columns)
    if not membership_df.empty:
        membership_df = membership_df.sort_values(
            by=["community_id", "pagerank", "total_stars", "topic"],
            ascending=[True, False, False, True],
        )

    community_df = pd.DataFrame(community_rows, columns=summary_columns)
    if not community_df.empty:
        community_df = community_df.sort_values(
            by=["size", "total_stars", "community_id"],
            ascending=[False, False, True],
        )

    communities_json = []
    for _, community_row in community_df.iterrows():
        community_id = int(community_row["community_id"])
        members = membership_df[membership_df["community_id"] == community_id]
        communities_json.append(
            {
                "community_id": community_id,
                "size": int(community_row["size"]),
                "internal_edges": int(community_row["internal_edges"]),
                "density": float(community_row["density"]),
                "total_stars": int(community_row["total_stars"]),
                "total_forks": int(community_row["total_forks"]),
                "total_repo_count": int(community_row["total_repo_count"]),
                "dominant_group": community_row["dominant_group"],
                "domain_purity": float(community_row["domain_purity"]),
                "domain_entropy": float(community_row["domain_entropy"]),
                "dominant_semantic_label": community_row["dominant_semantic_label"],
                "proposed_label": community_row["proposed_label"],
                "semantic_purity": float(community_row["semantic_purity"]),
                "semantic_entropy": float(community_row["semantic_entropy"]),
                "top_topics": community_row["top_topics"].split(", ")
                if community_row["top_topics"]
                else [],
                "members": members.to_dict(orient="records"),
            }
        )

    return membership_df, community_df, communities_json


def save_community_outputs(
    membership_df: pd.DataFrame,
    community_df: pd.DataFrame,
    communities_json: List[Dict],
    membership_output: str,
    summary_output: str,
    purity_output: str,
    json_output: str,
) -> None:
    """Persist community tables/files for downstream analysis."""
    membership_df.to_csv(membership_output, index=False, quoting=csv.QUOTE_MINIMAL)
    community_df.to_csv(summary_output, index=False, quoting=csv.QUOTE_MINIMAL)
    purity_columns = [
        "community_id",
        "size",
        "dominant_group",
        "domain_purity",
        "domain_entropy",
        "dominant_semantic_label",
        "proposed_label",
        "semantic_purity",
        "semantic_entropy",
        "top_topics",
    ]
    community_df[purity_columns].to_csv(purity_output, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(communities_json, f, ensure_ascii=False, indent=2)


def normalize_topic_name(topic: str) -> Set[str]:
    """Create normalized topic tokens for heuristic semantic labeling."""
    normalized = topic.lower().replace("_", "-")
    pieces = {normalized}
    pieces.update(part for part in re.split(r"[^a-z0-9]+", normalized) if part)
    return pieces


def infer_semantic_label(topic: str) -> str:
    """Infer a semantic subfield label from the topic name."""
    topic_tokens = normalize_topic_name(topic)
    best_label = "General AI"
    best_score = 0

    for label, keywords in SEMANTIC_LABEL_RULES.items():
        score = len(topic_tokens & keywords)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def score_semantic_labels(nodes: List[str], graph: nx.Graph) -> Counter:
    """Score semantic labels for a community using node importance."""
    scores: Counter = Counter()

    for node in nodes:
        semantic_label = infer_semantic_label(node)
        node_attrs = graph.nodes[node]
        pagerank = float(node_attrs.get("pagerank", 0.0))
        degree = float(graph.degree(node))
        weight = pagerank * 1000.0 + degree

        if semantic_label == "General AI":
            weight *= 0.25

        scores[semantic_label] += weight

    if not scores:
        scores["Unknown"] = 0.0
    return scores


def choose_proposed_label(dominant_group: str, dominant_semantic_label: str) -> str:
    """Produce a user-facing label for the community."""
    if dominant_semantic_label not in {"General AI", "Unknown"}:
        return dominant_semantic_label
    if dominant_group == "CV":
        return "General Computer Vision"
    if dominant_group == "NLP":
        return "General NLP / Language AI"
    if dominant_group == "Cross-domain":
        return "General AI / Cross-domain"
    return "General AI"


def compute_purity(counter: Counter) -> Tuple[str, float]:
    """Return dominant label and its share."""
    total = sum(counter.values())
    if total == 0:
        return "Unknown", 0.0

    dominant_label, dominant_count = counter.most_common(1)[0]
    return dominant_label, dominant_count / total


def compute_entropy(counter: Counter) -> float:
    """Compute Shannon entropy from a label counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def print_graph_stats(graph: nx.Graph) -> None:
    print("\nNetwork Summary")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")

    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    print(f"Density: {density:.6f}")

    if graph.number_of_nodes() <= 1:
        avg_path_length = 0.0
        print("Average path length: 0.000000 (graph too small)")
        return

    if nx.is_connected(graph):
        avg_path_length = nx.average_shortest_path_length(graph)
        print(f"Average path length: {avg_path_length:.6f} (full graph)")
    else:
        largest_cc_nodes = max(nx.connected_components(graph), key=len)
        largest_cc = graph.subgraph(largest_cc_nodes).copy()
        avg_path_length = nx.average_shortest_path_length(largest_cc)
        print(
            "Average path length: "
            f"{avg_path_length:.6f} (largest connected component, "
            f"{largest_cc.number_of_nodes()} nodes)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI topic backbone network from GitHub GraphQL")
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing GitHub token (default: GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--raw-output",
        default="raw_data.json",
        help="Path to save fetched and cleaned repository data as JSON",
    )
    parser.add_argument(
        "--gexf-output",
        default="ai_backbone_network.gexf",
        help="Path to save final GEXF file",
    )
    parser.add_argument(
        "--community-membership-output",
        default="community_membership.csv",
        help="Path to save topic-level community membership table as CSV",
    )
    parser.add_argument(
        "--community-summary-output",
        default="community_summary.csv",
        help="Path to save community-level summary table as CSV",
    )
    parser.add_argument(
        "--community-purity-output",
        default="community_purity.csv",
        help="Path to save community purity metrics as CSV",
    )
    parser.add_argument(
        "--communities-json-output",
        default="communities.json",
        help="Path to save full community structure and members as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env)
    if not token:
        raise RuntimeError(
            f"Missing GitHub token. Please set environment variable: {args.token_env}"
        )

    repo_data = build_dataset(token)
    with open(args.raw_output, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, ensure_ascii=False, indent=2)
    print(f"[save] cleaned repository data -> {args.raw_output}")

    topic_graph, extra_stats = build_backbone_graph(repo_data)
    membership_df, community_df, communities_json = build_community_outputs(topic_graph)
    save_community_outputs(
        membership_df=membership_df,
        community_df=community_df,
        communities_json=communities_json,
        membership_output=args.community_membership_output,
        summary_output=args.community_summary_output,
        purity_output=args.community_purity_output,
        json_output=args.communities_json_output,
    )
    nx.write_gexf(topic_graph, args.gexf_output)

    print(f"[community] detection method -> {extra_stats['community_method']}")
    print(f"[save] community membership csv -> {args.community_membership_output}")
    print(f"[save] community summary csv -> {args.community_summary_output}")
    print(f"[save] community purity csv -> {args.community_purity_output}")
    print(f"[save] communities json -> {args.communities_json_output}")
    print(f"[save] gephi gexf -> {args.gexf_output}")
    print(f"[elite] total elite users used in model: {extra_stats['elite_user_count']}")

    print_graph_stats(topic_graph)


if __name__ == "__main__":
    main()
