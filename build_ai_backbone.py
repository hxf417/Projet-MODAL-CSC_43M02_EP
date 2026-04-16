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
import json
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

    if topic_graph.number_of_edges() > 0:
        partition = community_louvain.best_partition(topic_graph, weight="weight")
    else:
        partition = {n: 0 for n in topic_graph.nodes()}

    for topic in topic_graph.nodes():
        pr_score = float(pagerank.get(topic, 0.0))
        bw_score = float(betweenness.get(topic, 0.0))
        topic_graph.nodes[topic]["PageRank"] = pr_score
        topic_graph.nodes[topic]["Betweenness"] = bw_score
        topic_graph.nodes[topic]["pagerank"] = pr_score
        topic_graph.nodes[topic]["betweenness"] = bw_score
        topic_graph.nodes[topic]["community"] = int(partition.get(topic, 0))

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

    return topic_graph, {"elite_user_count": len(elite_users)}


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
    nx.write_gexf(topic_graph, args.gexf_output)
    print(f"[save] gephi gexf -> {args.gexf_output}")
    print(f"[elite] total elite users used in model: {extra_stats['elite_user_count']}")

    print_graph_stats(topic_graph)


if __name__ == "__main__":
    main()
