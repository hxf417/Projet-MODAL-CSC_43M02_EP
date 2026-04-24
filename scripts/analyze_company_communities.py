#!/usr/bin/env python3
"""Analyze company-level communities from repository-level AI network.

Pipeline:
1) Load normalized repository data and repository graph with repo communities.
2) Resolve owner -> company (GitHub profile company if available; fallback to owner login).
3) Aggregate repo-repo edges into company-company edges.
4) Run Louvain on the company graph.
5) Export GEXF + CSV summaries, including small-company community signals.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import community as community_louvain
import networkx as nx
import pandas as pd
import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.5

OWNER_QUERY = """
query($login: String!) {
  organization(login: $login) {
    login
    name
    company
  }
  user(login: $login) {
    login
    name
    company
  }
}
"""

BASE_COLOR_PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def graphql_request(
    query: str,
    variables: Dict,
    headers: Dict[str, str],
    session: requests.Session,
) -> Dict:
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
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError("Unreachable retry logic")


def load_github_token(env_name: str) -> str:
    raw = os.getenv(env_name)
    if not raw:
        raise RuntimeError(
            f"Missing GitHub token. Please set environment variable: {env_name}"
        )
    token = raw.strip()
    if token.startswith("Bearer "):
        token = token[len("Bearer ") :].strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1].strip()
    if not token:
        raise RuntimeError(f"{env_name} is empty after trimming.")
    if any(ch.isspace() for ch in token):
        raise RuntimeError(f"{env_name} contains whitespace.")
    try:
        token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{env_name} contains non-ASCII characters.") from exc
    return token


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def canonicalize_company(raw_company: str) -> str:
    s = (raw_company or "").strip().lower()
    if not s:
        return ""
    s = s.replace("@", " ")
    s = re.sub(r"https?://", " ", s)
    s = re.sub(r"[^a-z0-9&+._\\-\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()

    suffixes = {
        "inc",
        "inc.",
        "llc",
        "ltd",
        "ltd.",
        "co",
        "corp",
        "corporation",
        "company",
        "gmbh",
        "sa",
        "sas",
        "plc",
        "ag",
    }
    parts = s.split()
    while parts and parts[-1] in suffixes:
        parts.pop()
    return " ".join(parts).strip()


def build_community_color_map(ids: Iterable[int]) -> Dict[int, Tuple[int, int, int]]:
    ids_sorted = sorted(set(ids))
    out: Dict[int, Tuple[int, int, int]] = {}
    for idx, cid in enumerate(ids_sorted):
        if idx < len(BASE_COLOR_PALETTE):
            out[cid] = BASE_COLOR_PALETTE[idx]
        else:
            hue = (idx * 0.61803398875) % 1.0
            sat = 0.63
            val = 0.92
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
            out[cid] = (int(r_f * 255), int(g_f * 255), int(b_f * 255))
    return out


def safe_eigenvector_centrality(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    out: Dict[str, float] = {}
    for comp_nodes in nx.connected_components(graph):
        sub = graph.subgraph(comp_nodes).copy()
        if sub.number_of_nodes() == 1:
            only = next(iter(sub.nodes()))
            out[only] = 1.0
            continue
        try:
            sub_eig = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except Exception:
            sub_eig = nx.eigenvector_centrality(sub, max_iter=2000, weight="weight")
        out.update(sub_eig)
    max_val = max(out.values()) if out else 1.0
    if max_val <= 0:
        return out
    for k in list(out.keys()):
        out[k] = float(out[k] / max_val)
    return out


def fetch_owner_company_map(
    owners: List[str],
    token: str,
    cache_path: str,
) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    missing = [o for o in owners if o not in cache]
    if not missing:
        return cache

    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()
    total = len(missing)
    for idx, owner in enumerate(missing, start=1):
        data = graphql_request(OWNER_QUERY, {"login": owner}, headers, session)
        org = data.get("organization")
        user = data.get("user")

        if org:
            owner_type = "Organization"
            name = org.get("name") or org.get("login") or owner
            company_raw = (org.get("company") or "").strip()
        elif user:
            owner_type = "User"
            name = user.get("name") or user.get("login") or owner
            company_raw = (user.get("company") or "").strip()
        else:
            owner_type = "Unknown"
            name = owner
            company_raw = ""

        cache[owner] = {
            "owner_login": owner,
            "owner_type": owner_type,
            "owner_name": name,
            "company_raw": company_raw,
            "company_canonical": canonicalize_company(company_raw),
        }
        if idx % 20 == 0 or idx == total:
            print(f"[owner] fetched {idx}/{total}")
        time.sleep(0.12)

    ensure_parent_dir(cache_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return cache


def owner_to_company_id(owner: str, owner_meta: Dict[str, Dict]) -> Tuple[str, str]:
    meta = owner_meta.get(owner, {})
    canonical = (meta.get("company_canonical") or "").strip()
    company_raw = (meta.get("company_raw") or "").strip()
    if canonical:
        return canonical, company_raw or canonical
    return owner.lower(), owner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build company-level communities from repository-level network"
    )
    parser.add_argument(
        "--repo-raw",
        default="data/raw/repo_raw_data_fork.json",
        help="Normalized repository raw JSON path",
    )
    parser.add_argument(
        "--repo-graph",
        default="outputs/graphs/repo/repository_backbone_fork.gexf",
        help="Repository network GEXF path (must contain Community attribute)",
    )
    parser.add_argument(
        "--company-graph",
        default="outputs/graphs/company/company_network.gexf",
        help="Output company network GEXF path",
    )
    parser.add_argument(
        "--company-nodes-csv",
        default="outputs/tables/company_nodes.csv",
        help="Output company node table CSV path",
    )
    parser.add_argument(
        "--company-communities-csv",
        default="outputs/tables/company_communities.csv",
        help="Output company community summary CSV path",
    )
    parser.add_argument(
        "--owner-cache",
        default="data/raw/owner_company_cache.json",
        help="Owner company profile cache JSON path",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="GitHub token env var (used when fetching owner profiles)",
    )
    parser.add_argument(
        "--fetch-owner-company",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch owner company metadata from GitHub profiles",
    )
    parser.add_argument(
        "--small-company-threshold",
        type=int,
        default=3,
        help="Small company if repo_count <= this threshold in sampled data",
    )
    parser.add_argument(
        "--min-company-edge-weight",
        type=float,
        default=0.02,
        help="Minimum normalized company edge weight",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution for company graph",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.repo_raw, "r", encoding="utf-8") as f:
        raw_repos = json.load(f)
    G_repo = nx.read_gexf(args.repo_graph)

    repo_raw_map = {r["nameWithOwner"]: r for r in raw_repos if "nameWithOwner" in r}

    # Restrict analysis to repositories that survived repo-level backbone graph.
    active_repos = [n for n in G_repo.nodes() if n in repo_raw_map]
    if not active_repos:
        raise RuntimeError("No overlap between repo_raw and repo_graph nodes.")

    owners = sorted({r.split("/", 1)[0] for r in active_repos})
    if args.fetch_owner_company:
        token = load_github_token(args.token_env)
        owner_meta = fetch_owner_company_map(
            owners=owners,
            token=token,
            cache_path=args.owner_cache,
        )
    else:
        owner_meta = {
            o: {
                "owner_login": o,
                "owner_type": "Unknown",
                "owner_name": o,
                "company_raw": "",
                "company_canonical": "",
            }
            for o in owners
        }

    repo_to_company: Dict[str, str] = {}
    company_label_map: Dict[str, str] = {}
    company_owner_set: Dict[str, set] = defaultdict(set)

    for repo in active_repos:
        owner = repo.split("/", 1)[0]
        company_id, company_label = owner_to_company_id(owner, owner_meta)
        repo_to_company[repo] = company_id
        company_label_map.setdefault(company_id, company_label)
        company_owner_set[company_id].add(owner)

    # Aggregate per-company node stats.
    company_repo_count = Counter()
    company_total_stars = Counter()
    company_domain_counter: Dict[str, Counter] = defaultdict(Counter)
    company_repo_community_counter: Dict[str, Counter] = defaultdict(Counter)

    for repo in active_repos:
        comp = repo_to_company[repo]
        raw = repo_raw_map[repo]
        company_repo_count[comp] += 1
        company_total_stars[comp] += int(raw.get("stargazerCount") or 0)
        for dm in raw.get("source_domains", []):
            company_domain_counter[comp][dm] += 1
        repo_comm = int(float(G_repo.nodes[repo].get("Community", 0)))
        company_repo_community_counter[comp][repo_comm] += 1

    # Aggregate company-company edges from repo-repo weighted edges.
    edge_weight_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_repo_pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    intra_weight_sum: Counter = Counter()

    for u, v, d in G_repo.edges(data=True):
        if u not in repo_to_company or v not in repo_to_company:
            continue
        cu = repo_to_company[u]
        cv = repo_to_company[v]
        w = float(d.get("weight", 0.0))
        if cu == cv:
            intra_weight_sum[cu] += w
            continue
        a, b = sorted((cu, cv))
        edge_weight_sum[(a, b)] += w
        edge_repo_pair_count[(a, b)] += 1

    G_company = nx.Graph()
    for comp in sorted(company_repo_count):
        repo_count = company_repo_count[comp]
        total_stars = company_total_stars[comp]
        domain_counts = company_domain_counter[comp]
        repo_comm_counts = company_repo_community_counter[comp]
        dominant_repo_comm = repo_comm_counts.most_common(1)[0][0] if repo_comm_counts else -1
        domain_group = (
            "Cross-domain"
            if ("computer-vision" in domain_counts and "nlp" in domain_counts)
            else ("CV" if "computer-vision" in domain_counts else ("NLP" if "nlp" in domain_counts else "Unknown"))
        )

        G_company.add_node(
            comp,
            label=company_label_map.get(comp, comp),
            RepoCount=int(repo_count),
            TotalStars=int(total_stars),
            OwnerCount=int(len(company_owner_set[comp])),
            SmallCompany=int(repo_count <= args.small_company_threshold),
            DominantRepoCommunity=int(dominant_repo_comm),
            DomainGroup=domain_group,
            IntraCompanyEdgeWeight=float(intra_weight_sum.get(comp, 0.0)),
            NodeSize=float(10.0 + 9.0 * math.sqrt(repo_count)),
        )

    for (a, b), w_raw in edge_weight_sum.items():
        # Normalize to reduce large-company size bias.
        norm = w_raw / math.sqrt(max(company_repo_count[a], 1) * max(company_repo_count[b], 1))
        if norm < args.min_company_edge_weight:
            continue
        G_company.add_edge(
            a,
            b,
            weight=float(norm),
            raw_weight=float(w_raw),
            repo_pair_count=int(edge_repo_pair_count[(a, b)]),
        )

    if G_company.number_of_edges() == 0:
        raise RuntimeError("Company graph has no edges after filtering. Lower min-company-edge-weight.")

    partition = community_louvain.best_partition(
        G_company,
        weight="weight",
        resolution=args.resolution,
    )
    modularity = community_louvain.modularity(partition, G_company, weight="weight")

    colors = build_community_color_map(partition.values())
    pagerank = nx.pagerank(G_company, weight="weight")
    eig = safe_eigenvector_centrality(G_company)

    dist_graph = G_company.copy()
    for _, _, d in dist_graph.edges(data=True):
        d["distance"] = 1.0 / max(float(d.get("weight", 0.0)), 1e-9)
    btw = nx.betweenness_centrality(dist_graph, weight="distance")

    for n in G_company.nodes():
        c = int(partition[n])
        r, g, b = colors[c]
        G_company.nodes[n]["CompanyCommunity"] = c
        G_company.nodes[n]["PageRank"] = float(pagerank.get(n, 0.0))
        G_company.nodes[n]["Eigenvector"] = float(eig.get(n, 0.0))
        G_company.nodes[n]["Betweenness"] = float(btw.get(n, 0.0))
        G_company.nodes[n]["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": 0.95},
            "size": float(G_company.nodes[n].get("NodeSize", 10.0)),
        }

    for u, v, d in G_company.edges(data=True):
        cu = partition[u]
        cv = partition[v]
        if cu == cv:
            r, g, b = colors[cu]
            alpha = 0.30
        else:
            r, g, b = (130, 130, 130)
            alpha = 0.15
        d["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": alpha},
            "thickness": float(0.8 + 7.0 * float(d.get("weight", 0.0))),
        }

    # CSV outputs.
    company_rows = []
    for n, d in G_company.nodes(data=True):
        company_rows.append(
            {
                "company_id": n,
                "company_label": d.get("label", n),
                "company_community": d.get("CompanyCommunity", -1),
                "repo_count": d.get("RepoCount", 0),
                "owner_count": d.get("OwnerCount", 0),
                "total_stars": d.get("TotalStars", 0),
                "small_company": d.get("SmallCompany", 0),
                "dominant_repo_community": d.get("DominantRepoCommunity", -1),
                "domain_group": d.get("DomainGroup", "Unknown"),
                "pagerank": d.get("PageRank", 0.0),
                "betweenness": d.get("Betweenness", 0.0),
                "eigenvector": d.get("Eigenvector", 0.0),
            }
        )
    company_df = pd.DataFrame(company_rows).sort_values(
        by=["company_community", "repo_count", "total_stars"],
        ascending=[True, False, False],
    )
    ensure_parent_dir(args.company_nodes_csv)
    company_df.to_csv(args.company_nodes_csv, index=False)

    com_rows = []
    for cc, group in company_df.groupby("company_community"):
        count = len(group)
        small_count = int(group["small_company"].sum())
        small_ratio = float(small_count / count) if count else 0.0
        top_companies = ", ".join(group.head(5)["company_label"].tolist())
        com_rows.append(
            {
                "company_community": int(cc),
                "company_count": int(count),
                "small_company_count": int(small_count),
                "small_company_ratio": float(round(small_ratio, 4)),
                "median_repo_count": float(group["repo_count"].median()),
                "median_stars": float(group["total_stars"].median()),
                "top_companies": top_companies,
            }
        )
    community_df = pd.DataFrame(com_rows).sort_values(
        by=["small_company_ratio", "company_count"],
        ascending=[False, False],
    )
    ensure_parent_dir(args.company_communities_csv)
    community_df.to_csv(args.company_communities_csv, index=False)

    ensure_parent_dir(args.company_graph)
    nx.write_gexf(G_company, args.company_graph)

    print("[save] company graph:", args.company_graph)
    print("[save] company nodes csv:", args.company_nodes_csv)
    print("[save] company communities csv:", args.company_communities_csv)
    print(
        "[summary] "
        f"companies={G_company.number_of_nodes()}, "
        f"edges={G_company.number_of_edges()}, "
        f"modularity={modularity:.4f}"
    )
    print("\nTop company communities by small-company ratio:")
    if not community_df.empty:
        print(community_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
