#!/usr/bin/env python3
"""Build an AI technology backbone network from GitHub GraphQL data.

Pipeline:
1) Fetch top-star repositories for topic:computer-vision and topic:nlp.
2) Fetch per-repository details (topics + mentionable users).
3) Build elite-user -> topic bipartite graph, then project to topic-topic.
4) Enrich node attributes and export Gephi-compatible GEXF.
5) Print core network statistics and top bridge technologies.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import community as community_louvain
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

BASE_COLOR_PALETTE = [
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (127, 127, 127),  # gray
    (188, 189, 34),   # olive
    (23, 190, 207),   # cyan
]


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


def build_community_color_map(partition: Dict[str, int]) -> Dict[int, Tuple[int, int, int]]:
    """Assign deterministic RGB colors to community ids."""
    community_ids = sorted(set(partition.values()))
    color_map: Dict[int, Tuple[int, int, int]] = {}

    for idx, cid in enumerate(community_ids):
        if idx < len(BASE_COLOR_PALETTE):
            color_map[cid] = BASE_COLOR_PALETTE[idx]
            continue
        hue = (idx * 0.61803398875) % 1.0
        sat = 0.62
        val = 0.92
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
        color_map[cid] = (int(r_f * 255), int(g_f * 255), int(b_f * 255))
    return color_map


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


def build_backbone_graph(
    repo_data: List[Dict],
    min_shared_elite_users: int,
    generic_topic_repo_ratio: float,
    top_k_per_topic: int,
    k_core_k: int,
) -> Tuple[nx.Graph, Dict[str, int]]:
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

    total_repos = max(len(repo_data), 1)
    topic_repo_count = {
        topic: int(stats.get("repo_count", 0)) for topic, stats in topic_stats.items()
    }

    # Remove overly generic topics that connect everything and hide community structure.
    generic_threshold = max(8, int(total_repos * generic_topic_repo_ratio))
    generic_topics = {
        t for t, c in topic_repo_count.items() if c >= generic_threshold
    }
    if generic_topics:
        topic_graph.remove_nodes_from([n for n in generic_topics if n in topic_graph])

    # Shared elite-user count + association strength (normalized) for each edge.
    for u, v, data in topic_graph.edges(data=True):
        shared = int(data.get("weight", 0))
        data["shared_elite_users"] = shared
        denom = math.sqrt(
            max(topic_repo_count.get(u, 1), 1) * max(topic_repo_count.get(v, 1), 1)
        )
        data["association_strength"] = float(shared / denom)

    # Keep only edges with enough support.
    weak_edges = [
        (u, v)
        for u, v, d in topic_graph.edges(data=True)
        if d.get("shared_elite_users", 0) < min_shared_elite_users
    ]
    topic_graph.remove_edges_from(weak_edges)
    topic_graph.remove_nodes_from(list(nx.isolates(topic_graph)))

    # Top-k sparsification per topic to reveal clusters (symmetrized keep set).
    keep_edges: Set[Tuple[str, str]] = set()
    for node in topic_graph.nodes():
        incident = []
        for nbr in topic_graph.neighbors(node):
            edge = topic_graph[node][nbr]
            incident.append(
                (
                    edge.get("association_strength", 0.0),
                    edge.get("shared_elite_users", 0),
                    tuple(sorted((node, nbr))),
                )
            )
        incident.sort(reverse=True)
        for _, _, edge_key in incident[:top_k_per_topic]:
            keep_edges.add(edge_key)

    edges_to_remove = []
    for u, v in topic_graph.edges():
        edge_key = tuple(sorted((u, v)))
        if edge_key not in keep_edges:
            edges_to_remove.append((u, v))
    topic_graph.remove_edges_from(edges_to_remove)
    topic_graph.remove_nodes_from(list(nx.isolates(topic_graph)))

    # k-core cleanup: drop weakly connected fringe nodes.
    if topic_graph.number_of_nodes() > 0 and k_core_k > 0:
        topic_graph = nx.k_core(topic_graph, k=k_core_k)

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
        topic_graph.nodes[topic]["NodeSize"] = float(8.0 + 6.0 * math.log10(total_stars + 1))
        topic_graph.nodes[topic]["Forks"] = total_forks
        topic_graph.nodes[topic]["RepoCount"] = repo_count

        # Lowercase aliases for easier programmatic use.
        topic_graph.nodes[topic]["total_stars"] = total_stars
        topic_graph.nodes[topic]["total_forks"] = total_forks
        topic_graph.nodes[topic]["group"] = group
        topic_graph.nodes[topic]["language"] = dominant_language
        topic_graph.nodes[topic]["repo_count"] = repo_count

    pagerank = (
        nx.pagerank(topic_graph, weight="association_strength")
        if topic_graph.number_of_nodes()
        else {}
    )

    distance_graph = topic_graph.copy()
    for u, v, data in distance_graph.edges(data=True):
        strength = data.get("association_strength", 0.0)
        data["distance"] = 1.0 / max(float(strength), 1e-9)
    betweenness = (
        nx.betweenness_centrality(distance_graph, weight="distance")
        if topic_graph.number_of_nodes()
        else {}
    )

    modularity = 0.0
    if topic_graph.number_of_edges() > 0:
        partition = community_louvain.best_partition(
            topic_graph,
            weight="association_strength",
            resolution=1.15,
        )
        modularity = community_louvain.modularity(
            partition,
            topic_graph,
            weight="association_strength",
        )
    else:
        partition = {n: 0 for n in topic_graph.nodes()}

    for topic in topic_graph.nodes():
        pr_score = float(pagerank.get(topic, 0.0))
        bw_score = float(betweenness.get(topic, 0.0))
        topic_graph.nodes[topic]["PageRank"] = pr_score
        topic_graph.nodes[topic]["Betweenness"] = bw_score
        topic_graph.nodes[topic]["pagerank"] = pr_score
        topic_graph.nodes[topic]["betweenness"] = bw_score
        topic_graph.nodes[topic]["Community"] = int(partition.get(topic, 0))
        topic_graph.nodes[topic]["community"] = int(partition.get(topic, 0))

    community_colors = build_community_color_map(partition)
    for topic in topic_graph.nodes():
        cid = int(partition.get(topic, 0))
        r, g, b = community_colors[cid]
        topic_graph.nodes[topic]["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": 0.95},
            "size": float(topic_graph.nodes[topic].get("NodeSize", 10.0)),
        }

    for u, v, edge_data in topic_graph.edges(data=True):
        cu = int(partition.get(u, 0))
        cv = int(partition.get(v, 0))
        if cu == cv:
            r, g, b = community_colors[cu]
            alpha = 0.35
        else:
            r, g, b = (140, 140, 140)
            alpha = 0.20
        edge_data["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": alpha},
            "thickness": float(1.0 + 8.0 * edge_data.get("association_strength", 0.0)),
        }

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
        "generic_topics_removed": len(generic_topics),
        "min_shared_elite_users": min_shared_elite_users,
        "top_k_per_topic": top_k_per_topic,
        "k_core_k": k_core_k,
        "generic_threshold": generic_threshold,
        "communities": len(set(partition.values())) if partition else 0,
        "modularity": float(modularity),
    }


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
        "--min-shared-users",
        type=int,
        default=3,
        help="Minimum shared elite developers for keeping a topic-topic edge",
    )
    parser.add_argument(
        "--generic-ratio",
        type=float,
        default=0.18,
        help="Drop topics that appear in at least this ratio of repositories",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Keep top-k strongest edges per topic by association strength",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=2,
        help="Apply k-core cleanup after edge pruning (0 disables)",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing raw JSON file instead of fetching from GitHub API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reuse_raw and os.path.exists(args.raw_output):
        with open(args.raw_output, "r", encoding="utf-8") as f:
            repo_data = json.load(f)
        print(f"[load] reused existing raw data <- {args.raw_output}")
    else:
        token = os.getenv(args.token_env)
        if not token:
            raise RuntimeError(
                f"Missing GitHub token. Please set environment variable: {args.token_env}"
            )
        repo_data = build_dataset(token)
        with open(args.raw_output, "w", encoding="utf-8") as f:
            json.dump(repo_data, f, ensure_ascii=False, indent=2)
        print(f"[save] cleaned repository data -> {args.raw_output}")

    topic_graph, extra_stats = build_backbone_graph(
        repo_data=repo_data,
        min_shared_elite_users=args.min_shared_users,
        generic_topic_repo_ratio=args.generic_ratio,
        top_k_per_topic=args.top_k,
        k_core_k=args.k_core,
    )
    nx.write_gexf(topic_graph, args.gexf_output)
    print(f"[save] gephi gexf -> {args.gexf_output}")
    print(f"[elite] total elite users used in model: {extra_stats['elite_user_count']}")
    print(
        "[prune] "
        f"generic_topics_removed={extra_stats['generic_topics_removed']}, "
        f"generic_threshold={extra_stats['generic_threshold']}, "
        f"min_shared_elite_users={extra_stats['min_shared_elite_users']}, "
        f"top_k_per_topic={extra_stats['top_k_per_topic']}, "
        f"k_core_k={extra_stats['k_core_k']}"
    )
    print(
        "[community] "
        f"communities={extra_stats['communities']}, "
        f"modularity={extra_stats['modularity']:.4f}"
    )

    print_graph_stats(topic_graph)


if __name__ == "__main__":
    main()
