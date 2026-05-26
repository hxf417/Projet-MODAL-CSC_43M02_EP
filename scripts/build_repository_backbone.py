#!/usr/bin/env python3
"""Build a repository-level AI backbone network from GitHub GraphQL data.

Network design:
- Node: repository
- Edge weight:
  w = 0.6 * fork_overlap
      + 0.2 * readme_semantic_similarity
      + 0.2 * dependency_overlap
- Backbone: edge threshold + top-k sparsification + k-core cleanup
- Community: Louvain
- Centrality: weighted PageRank / Betweenness / Eigenvector
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import community as community_louvain
import networkx as nx
import pandas as pd
import requests

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
DOMAIN_CV = "computer-vision"
DOMAIN_NLP = "nlp"
DOMAINS = (DOMAIN_CV, DOMAIN_NLP)

REPO_LIMIT_PER_DOMAIN = 100
REPOS_PAGE_SIZE = 50
TOPICS_LIMIT = 100
FORKS_LIMIT = 100
REQUEST_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 45

README_MAX_CHARS = 60_000

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

DETAIL_QUERY = """
query(
  $owner: String!,
  $name: String!,
  $topicsFirst: Int!,
  $forksFirst: Int!
) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    stargazerCount
    forkCount
    isFork
    primaryLanguage { name }
    parent { nameWithOwner }
    repositoryTopics(first: $topicsFirst) {
      nodes { topic { name } }
    }
    readmeMd: object(expression: "HEAD:README.md") { ... on Blob { text } }
    readmeLower: object(expression: "HEAD:readme.md") { ... on Blob { text } }
    readmeRst: object(expression: "HEAD:README.rst") { ... on Blob { text } }
    readmeTxt: object(expression: "HEAD:README.txt") { ... on Blob { text } }
    readme: object(expression: "HEAD:README") { ... on Blob { text } }

    pkgJson: object(expression: "HEAD:package.json") { ... on Blob { text } }
    reqTxt: object(expression: "HEAD:requirements.txt") { ... on Blob { text } }
    pyprojectToml: object(expression: "HEAD:pyproject.toml") { ... on Blob { text } }
    goMod: object(expression: "HEAD:go.mod") { ... on Blob { text } }
    cargoToml: object(expression: "HEAD:Cargo.toml") { ... on Blob { text } }
    pomXml: object(expression: "HEAD:pom.xml") { ... on Blob { text } }
    gemfile: object(expression: "HEAD:Gemfile") { ... on Blob { text } }
    envYml: object(expression: "HEAD:environment.yml") { ... on Blob { text } }
    forks(first: $forksFirst) {
      totalCount
      nodes {
        nameWithOwner
        isArchived
        pushedAt
        stargazerCount
        owner { login }
      }
    }
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

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\\-]{1,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "your",
    "you",
    "are",
    "our",
    "their",
    "using",
    "use",
    "used",
    "into",
    "about",
    "have",
    "has",
    "was",
    "were",
    "will",
    "can",
    "all",
    "any",
    "new",
    "more",
    "most",
    "also",
    "not",
    "but",
    "via",
    "http",
    "https",
    "www",
    "github",
    "project",
    "repository",
    "code",
    "model",
    "models",
    "learning",
    "machine",
    "deep",
    "ai",
}

DEP_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/:+-]*$")


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

    # Remove accidental shell quoting in exported value.
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1].strip()

    if not token:
        raise RuntimeError(f"{env_name} is empty after trimming.")

    if any(ch.isspace() for ch in token):
        raise RuntimeError(
            f"{env_name} contains whitespace. Export raw token only, without spaces/newlines."
        )

    try:
        token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{env_name} contains non-ASCII characters. "
            "Please re-copy token from GitHub and avoid Chinese quotes/full-width characters."
        ) from exc

    if len(token) < 20:
        raise RuntimeError(
            f"{env_name} looks too short ({len(token)} chars). "
            "Please verify you exported the full token."
        )
    return token


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def is_filtered_login(login: str) -> bool:
    lower = login.lower()
    return "[bot]" in lower or lower == "web-flow"


def clean_logins(logins: Iterable[str]) -> List[str]:
    dedup = []
    seen = set()
    for login in logins:
        if not login:
            continue
        if is_filtered_login(login):
            continue
        if login in seen:
            continue
        seen.add(login)
        dedup.append(login)
    return dedup


def safe_blob_text(blob_obj: Optional[Dict]) -> str:
    if not blob_obj or not isinstance(blob_obj, dict):
        return ""
    text = blob_obj.get("text") or ""
    return text if isinstance(text, str) else ""


def pick_readme_text(repo: Dict) -> str:
    for key in ("readmeMd", "readmeLower", "readmeRst", "readmeTxt", "readme"):
        text = safe_blob_text(repo.get(key))
        if text.strip():
            return text[:README_MAX_CHARS]
    if isinstance(repo.get("readme_text"), str):
        return repo["readme_text"][:README_MAX_CHARS]
    return ""


def normalize_dep_name(raw: str) -> Optional[str]:
    name = raw.strip().lower()
    if not name:
        return None
    if "#egg=" in name:
        name = name.split("#egg=", 1)[1]
    name = name.split("[", 1)[0]
    name = re.split(r"[<>=!~;,@\\s]+", name, maxsplit=1)[0].strip()
    if not name:
        return None
    if not DEP_NAME_RE.match(name):
        return None
    if len(name) <= 1:
        return None
    return name


def parse_package_json(text: str) -> Set[str]:
    deps = set()
    try:
        data = json.loads(text)
    except Exception:
        return deps
    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        value = data.get(section)
        if isinstance(value, dict):
            for key in value.keys():
                norm = normalize_dep_name(str(key))
                if norm:
                    deps.add(norm)
    return deps


def parse_requirements_txt(text: str) -> Set[str]:
    deps = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement"):
            continue
        norm = normalize_dep_name(line)
        if norm:
            deps.add(norm)
    return deps


def parse_pyproject_toml(text: str) -> Set[str]:
    deps = set()
    if not tomllib:
        return deps
    try:
        data = tomllib.loads(text)
    except Exception:
        return deps

    project = data.get("project", {})
    if isinstance(project, dict):
        for item in project.get("dependencies", []) or []:
            norm = normalize_dep_name(str(item))
            if norm:
                deps.add(norm)
        optional = project.get("optional-dependencies", {}) or {}
        if isinstance(optional, dict):
            for dep_list in optional.values():
                if isinstance(dep_list, list):
                    for item in dep_list:
                        norm = normalize_dep_name(str(item))
                        if norm:
                            deps.add(norm)

    tool = data.get("tool", {})
    poetry = (tool.get("poetry", {}) if isinstance(tool, dict) else {}) or {}
    poetry_deps = poetry.get("dependencies", {})
    if isinstance(poetry_deps, dict):
        for key in poetry_deps.keys():
            norm = normalize_dep_name(str(key))
            if norm and norm != "python":
                deps.add(norm)
    return deps


def parse_go_mod(text: str) -> Set[str]:
    deps = set()
    in_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("require ("):
            in_block = True
            continue
        if in_block and line == ")":
            in_block = False
            continue
        if in_block:
            parts = line.split()
            if parts:
                norm = normalize_dep_name(parts[0])
                if norm:
                    deps.add(norm)
            continue
        if line.startswith("require "):
            parts = line[len("require ") :].split()
            if parts:
                norm = normalize_dep_name(parts[0])
                if norm:
                    deps.add(norm)
    return deps


def parse_cargo_toml(text: str) -> Set[str]:
    deps = set()
    if tomllib:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = {}
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            sec = data.get(section, {})
            if isinstance(sec, dict):
                for key in sec.keys():
                    norm = normalize_dep_name(str(key))
                    if norm:
                        deps.add(norm)
    return deps


def parse_pom_xml(text: str) -> Set[str]:
    deps = set()
    try:
        root = ET.fromstring(text)
    except Exception:
        root = None
    if root is not None:
        for dep in root.findall(".//{*}dependency"):
            group_id = dep.findtext("{*}groupId", default="").strip()
            artifact_id = dep.findtext("{*}artifactId", default="").strip()
            if group_id and artifact_id:
                norm = normalize_dep_name(f"{group_id}:{artifact_id}")
            else:
                norm = normalize_dep_name(artifact_id or group_id)
            if norm:
                deps.add(norm)
        return deps

    for artifact in re.findall(r"<artifactId>([^<]+)</artifactId>", text):
        norm = normalize_dep_name(artifact)
        if norm:
            deps.add(norm)
    return deps


def parse_gemfile(text: str) -> Set[str]:
    deps = set()
    for name in re.findall(r"^\\s*gem\\s+[\"']([^\"']+)[\"']", text, flags=re.MULTILINE):
        norm = normalize_dep_name(name)
        if norm:
            deps.add(norm)
    return deps


def parse_environment_yml(text: str) -> Set[str]:
    deps = set()
    in_pip_block = False
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line == "dependencies:":
            in_pip_block = False
            continue
        if line == "- pip:":
            in_pip_block = True
            continue
        if line.startswith("- "):
            item = line[2:].strip()
            if item.endswith(":"):
                continue
            norm = normalize_dep_name(item)
            if norm:
                deps.add(norm)
            continue
        if in_pip_block and indent >= 4 and line.startswith("- "):
            item = line[2:].strip()
            norm = normalize_dep_name(item)
            if norm:
                deps.add(norm)
    return deps


def extract_dependencies(repo: Dict) -> List[str]:
    if isinstance(repo.get("dependencies"), list):
        return sorted(set(str(x) for x in repo["dependencies"] if str(x).strip()))

    dep_set: Set[str] = set()
    text_fields = {
        "pkgJson": safe_blob_text(repo.get("pkgJson")),
        "reqTxt": safe_blob_text(repo.get("reqTxt")),
        "pyprojectToml": safe_blob_text(repo.get("pyprojectToml")),
        "goMod": safe_blob_text(repo.get("goMod")),
        "cargoToml": safe_blob_text(repo.get("cargoToml")),
        "pomXml": safe_blob_text(repo.get("pomXml")),
        "gemfile": safe_blob_text(repo.get("gemfile")),
        "envYml": safe_blob_text(repo.get("envYml")),
    }
    if text_fields["pkgJson"]:
        dep_set.update(parse_package_json(text_fields["pkgJson"]))
    if text_fields["reqTxt"]:
        dep_set.update(parse_requirements_txt(text_fields["reqTxt"]))
    if text_fields["pyprojectToml"]:
        dep_set.update(parse_pyproject_toml(text_fields["pyprojectToml"]))
    if text_fields["goMod"]:
        dep_set.update(parse_go_mod(text_fields["goMod"]))
    if text_fields["cargoToml"]:
        dep_set.update(parse_cargo_toml(text_fields["cargoToml"]))
    if text_fields["pomXml"]:
        dep_set.update(parse_pom_xml(text_fields["pomXml"]))
    if text_fields["gemfile"]:
        dep_set.update(parse_gemfile(text_fields["gemfile"]))
    if text_fields["envYml"]:
        dep_set.update(parse_environment_yml(text_fields["envYml"]))
    return sorted(dep_set)


def extract_topics(repo: Dict) -> List[str]:
    if isinstance(repo.get("topics"), list):
        return sorted(set(str(x) for x in repo["topics"] if str(x).strip()))
    nodes = ((repo.get("repositoryTopics") or {}).get("nodes") or [])
    topics = []
    for node in nodes:
        topic = (node.get("topic") or {}).get("name")
        if topic:
            topics.append(topic)
    return sorted(set(topics))


def extract_forker_owners(repo: Dict) -> List[str]:
    if isinstance(repo.get("forker_owners"), list):
        return clean_logins([str(x) for x in repo["forker_owners"]])

    owners = []
    forks_obj = repo.get("forks")
    if isinstance(forks_obj, dict):
        for node in forks_obj.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            if node.get("isArchived") is True:
                continue
            owner = node.get("owner") or {}
            login = (owner.get("login") or "").strip()
            if login:
                owners.append(login)
    return clean_logins(owners)


def extract_primary_language(repo: Dict) -> str:
    lang = repo.get("primaryLanguage")
    if isinstance(lang, str):
        return lang or "Unknown"
    if isinstance(lang, dict):
        return lang.get("name") or "Unknown"
    return "Unknown"


def normalize_loaded_repo_record(repo: Dict) -> Dict:
    name_with_owner = repo.get("nameWithOwner") or repo.get("full_name")
    if not name_with_owner:
        raise ValueError("Repository record missing nameWithOwner/full_name")
    source_domains = repo.get("source_domains") or []
    if isinstance(source_domains, str):
        source_domains = [source_domains]

    normalized = {
        "nameWithOwner": name_with_owner,
        "stargazerCount": int(repo.get("stargazerCount") or 0),
        "forkCount": int(repo.get("forkCount") or 0),
        "isFork": bool(repo.get("isFork") or False),
        "parentNameWithOwner": (
            ((repo.get("parent") or {}).get("nameWithOwner"))
            if isinstance(repo.get("parent"), dict)
            else (repo.get("parentNameWithOwner") or "")
        ),
        "primaryLanguage": extract_primary_language(repo),
        "source_domains": sorted(set(source_domains)),
        "topics": extract_topics(repo),
        "forker_owners": extract_forker_owners(repo),
        "readme_text": pick_readme_text(repo),
        "dependencies": extract_dependencies(repo),
    }
    return normalized


def fetch_top_repositories(
    domain: str,
    limit: int,
    headers: Dict[str, str],
    session: requests.Session,
) -> List[Dict]:
    repos: List[Dict] = []
    cursor = None
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
        nodes = search_data.get("nodes") or []
        repos.extend(nodes)
        page_info = search_data["pageInfo"]
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        time.sleep(0.2)
    return repos[:limit]


def fetch_repository_details(
    name_with_owner: str,
    headers: Dict[str, str],
    session: requests.Session,
) -> Dict:
    owner, name = name_with_owner.split("/", 1)
    data = graphql_request(
        DETAIL_QUERY,
        {
            "owner": owner,
            "name": name,
            "topicsFirst": TOPICS_LIMIT,
            "forksFirst": FORKS_LIMIT,
        },
        headers,
        session,
    )
    return data.get("repository") or {}


def build_dataset_from_github(token: str, per_domain_limit: int) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()

    repos_by_domain: Dict[str, List[Dict]] = {}
    for domain in DOMAINS:
        print(f"[list] fetching top {per_domain_limit} repos for {domain}")
        repos = fetch_top_repositories(
            domain=domain,
            limit=per_domain_limit,
            headers=headers,
            session=session,
        )
        repos_by_domain[domain] = repos
        print(f"[list] got {len(repos)} repos for {domain}")

    merged_index: Dict[str, Dict] = {}
    for domain, repos in repos_by_domain.items():
        for repo in repos:
            key = repo["nameWithOwner"]
            if key not in merged_index:
                merged_index[key] = {
                    "nameWithOwner": key,
                    "stargazerCount": int(repo.get("stargazerCount") or 0),
                    "forkCount": int(repo.get("forkCount") or 0),
                    "primaryLanguage": extract_primary_language(repo),
                    "source_domains": set(),
                }
            merged_index[key]["source_domains"].add(domain)

    print(
        f"[merge] requested {per_domain_limit * 2} repos, unique={len(merged_index)}"
    )

    normalized_records: List[Dict] = []
    total = len(merged_index)
    for idx, (name_with_owner, base) in enumerate(merged_index.items(), start=1):
        detail = fetch_repository_details(name_with_owner, headers, session)
        merged = {
            **detail,
            "nameWithOwner": base["nameWithOwner"],
            "stargazerCount": base["stargazerCount"],
            "forkCount": base["forkCount"],
            "primaryLanguage": base["primaryLanguage"],
            "source_domains": sorted(base["source_domains"]),
        }
        normalized_records.append(normalize_loaded_repo_record(merged))
        if idx % 20 == 0 or idx == total:
            print(f"[detail] fetched {idx}/{total}")
        time.sleep(0.15)

    return normalized_records


def normalize_loaded_dataset(raw_records: List[Dict]) -> List[Dict]:
    normalized = []
    for repo in raw_records:
        try:
            normalized.append(normalize_loaded_repo_record(repo))
        except Exception:
            continue
    return normalized


def preprocess_readme(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\\[[^\\]]*\\]\\([^\\)]*\\)", " ", text)
    text = re.sub(r"\\[[^\\]]*\\]\\([^\\)]*\\)", " ", text)
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


def build_readme_vectors(
    readme_tokens: List[List[str]],
    version: str,
) -> Tuple[List[Dict[str, float]], List[float]]:
    if version == "v2":
        return build_tfidf_vectors(readme_tokens)
    if version == "v1":
        # Legacy-compatible fallback: keep the same sparse cosine pipeline but
        # disable max document-frequency filtering.
        return build_tfidf_vectors(readme_tokens, max_df_ratio=1.0)
    raise ValueError(f"Unsupported README similarity version: {version}")


def cosine_similarity_sparse(
    vec_a: Dict[str, float],
    norm_a: float,
    vec_b: Dict[str, float],
    norm_b: float,
) -> float:
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
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


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    inter = set_a & set_b
    return len(inter) / len(union)


def normalized_weight_config(
    w_fork: float,
    w_readme: float,
    w_dep: float,
) -> Dict[str, float]:
    weights = {
        "fork": float(w_fork),
        "readme": float(w_readme),
        "dep": float(w_dep),
    }
    if any(v < 0 for v in weights.values()):
        raise ValueError("Similarity weights must be non-negative.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one similarity weight must be > 0.")
    return {k: v / total for k, v in weights.items()}


def build_repository_similarity_graph(
    repo_data: List[Dict],
    pre_min_weight: float,
    weight_fork: float,
    weight_readme: float,
    weight_dep: float,
    readme_similarity_version: str,
) -> Tuple[nx.Graph, Dict[str, float]]:
    graph = nx.Graph()
    weights = normalized_weight_config(
        w_fork=weight_fork,
        w_readme=weight_readme,
        w_dep=weight_dep,
    )

    readme_tokens = [preprocess_readme(repo.get("readme_text", "")) for repo in repo_data]
    tfidf_vectors, tfidf_norms = build_readme_vectors(
        readme_tokens,
        version=readme_similarity_version,
    )

    forker_sets = [set(repo.get("forker_owners", [])) for repo in repo_data]
    dependency_sets = [set(repo.get("dependencies", [])) for repo in repo_data]

    for idx, repo in enumerate(repo_data):
        domains = set(repo.get("source_domains", []))
        if DOMAIN_CV in domains and DOMAIN_NLP in domains:
            domain_group = "Cross-domain"
        elif DOMAIN_CV in domains:
            domain_group = "CV"
        elif DOMAIN_NLP in domains:
            domain_group = "NLP"
        else:
            domain_group = "Unknown"

        stars = int(repo.get("stargazerCount") or 0)
        graph.add_node(
            repo["nameWithOwner"],
            Stars=stars,
            Forks=int(repo.get("forkCount") or 0),
            Language=repo.get("primaryLanguage") or "Unknown",
            DomainGroup=domain_group,
            Topics="|".join(repo.get("topics", [])),
            TopicCount=int(len(repo.get("topics", []))),
            ForkerCount=int(len(forker_sets[idx])),
            DependencyCount=int(len(dependency_sets[idx])),
            ReadmeLength=int(len(repo.get("readme_text", ""))),
            IsFork=int(bool(repo.get("isFork") or False)),
            ParentRepo=repo.get("parentNameWithOwner") or "",
            NodeSize=float(8.0 + 6.0 * math.log10(stars + 1)),
        )

    pair_count = 0
    nonzero_fork = 0
    nonzero_readme = 0
    nonzero_dep = 0
    nonzero_weight = 0

    for i, j in combinations(range(len(repo_data)), 2):
        repo_i = repo_data[i]
        repo_j = repo_data[j]
        pair_count += 1

        fork_j = jaccard_similarity(forker_sets[i], forker_sets[j])
        readme_sim = cosine_similarity_sparse(
            tfidf_vectors[i], tfidf_norms[i], tfidf_vectors[j], tfidf_norms[j]
        )
        dep_j = jaccard_similarity(dependency_sets[i], dependency_sets[j])
        if fork_j > 0:
            nonzero_fork += 1
        if readme_sim > 0:
            nonzero_readme += 1
        if dep_j > 0:
            nonzero_dep += 1
        weight = (
            weights["fork"] * fork_j
            + weights["readme"] * readme_sim
            + weights["dep"] * dep_j
        )

        if weight < pre_min_weight:
            continue
        nonzero_weight += 1

        graph.add_edge(
            repo_i["nameWithOwner"],
            repo_j["nameWithOwner"],
            weight=float(weight),
            fork_overlap=float(fork_j),
            readme_similarity=float(readme_sim),
            dependency_overlap=float(dep_j),
        )

    stats = {
        "pairs": float(pair_count),
        "fork_nonzero_ratio": float(nonzero_fork / pair_count) if pair_count else 0.0,
        "readme_nonzero_ratio": float(nonzero_readme / pair_count) if pair_count else 0.0,
        "dep_nonzero_ratio": float(nonzero_dep / pair_count) if pair_count else 0.0,
        "edge_keep_ratio_pre": float(nonzero_weight / pair_count) if pair_count else 0.0,
        "weight_fork": weights["fork"],
        "weight_readme": weights["readme"],
        "weight_dep": weights["dep"],
        "readme_similarity_version": readme_similarity_version,
    }
    return graph, stats


def apply_backbone_filter(
    graph: nx.Graph,
    min_weight: float,
    top_k: int,
    k_core_k: int,
) -> nx.Graph:
    filtered = graph.copy()

    filtered.remove_edges_from(
        [(u, v) for u, v, d in filtered.edges(data=True) if d.get("weight", 0.0) < min_weight]
    )
    filtered.remove_nodes_from(list(nx.isolates(filtered)))

    if top_k > 0 and filtered.number_of_edges() > 0:
        keep_edges: Set[Tuple[str, str]] = set()
        for node in filtered.nodes():
            scored = []
            for nbr in filtered.neighbors(node):
                w = float(filtered[node][nbr].get("weight", 0.0))
                scored.append((w, tuple(sorted((node, nbr)))))
            scored.sort(reverse=True)
            for _, edge_key in scored[:top_k]:
                keep_edges.add(edge_key)

        drop_edges = []
        for u, v in filtered.edges():
            edge_key = tuple(sorted((u, v)))
            if edge_key not in keep_edges:
                drop_edges.append((u, v))
        filtered.remove_edges_from(drop_edges)
        filtered.remove_nodes_from(list(nx.isolates(filtered)))

    if k_core_k > 0 and filtered.number_of_nodes() > 0:
        filtered = nx.k_core(filtered, k=k_core_k)
    return filtered


def quick_modularity_estimate(graph: nx.Graph, resolution: float) -> float:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return 0.0
    try:
        partition = community_louvain.best_partition(
            graph,
            weight="weight",
            resolution=resolution,
        )
        return float(community_louvain.modularity(partition, graph, weight="weight"))
    except Exception:
        return 0.0


def auto_relax_backbone(
    base_graph: nx.Graph,
    min_weight: float,
    top_k: int,
    k_core_k: int,
    resolution: float,
    target_min_nodes: int,
) -> Tuple[nx.Graph, Dict[str, float]]:
    candidates: List[Tuple[float, int, int]] = []
    current = (float(min_weight), int(top_k), int(k_core_k))
    candidates.append(current)

    # Gradually relax constraints so the graph does not collapse to a tiny clique.
    for _ in range(10):
        mw, tk, kc = current
        if kc > 0:
            kc -= 1
        elif mw > 0.02:
            mw = max(0.02, round(mw * 0.8, 4))
        elif tk > 0:
            tk = min(30, tk + 4)
        else:
            break
        current = (mw, tk, kc)
        candidates.append(current)

    candidates.extend(
        [
            (0.06, max(top_k, 12), 1),
            (0.05, max(top_k, 16), 1),
            (0.04, 0, 0),
            (0.03, 0, 0),
            (0.02, 0, 0),
        ]
    )

    unique_candidates: List[Tuple[float, int, int]] = []
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        unique_candidates.append(c)

    best_graph = nx.Graph()
    best_meta = {
        "min_weight": min_weight,
        "top_k": top_k,
        "k_core": k_core_k,
        "modularity": 0.0,
        "nodes": 0.0,
        "edges": 0.0,
    }
    best_score = -1e18

    for mw, tk, kc in unique_candidates:
        g = apply_backbone_filter(base_graph, min_weight=mw, top_k=tk, k_core_k=kc)
        nodes = g.number_of_nodes()
        edges = g.number_of_edges()
        if nodes == 0 or edges == 0:
            score = -1e12 + nodes
            modularity = 0.0
        else:
            modularity = quick_modularity_estimate(g, resolution=resolution)
            # Prioritize meeting target node size, then modularity.
            meets_target = 1.0 if nodes >= target_min_nodes else 0.0
            score = (
                10_000.0 * meets_target
                + 250.0 * modularity
                + 3.0 * nodes
                + 0.25 * edges
            )
        if score > best_score:
            best_score = score
            best_graph = g
            best_meta = {
                "min_weight": mw,
                "top_k": float(tk),
                "k_core": float(kc),
                "modularity": modularity,
                "nodes": float(nodes),
                "edges": float(edges),
            }

    return best_graph, best_meta


def build_community_color_map(partition: Dict[str, int]) -> Dict[int, Tuple[int, int, int]]:
    community_ids = sorted(set(partition.values()))
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for idx, cid in enumerate(community_ids):
        if idx < len(BASE_COLOR_PALETTE):
            color_map[cid] = BASE_COLOR_PALETTE[idx]
        else:
            hue = (idx * 0.61803398875) % 1.0
            sat = 0.65
            val = 0.92
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
            color_map[cid] = (int(r_f * 255), int(g_f * 255), int(b_f * 255))
    return color_map


def annotate_graph_metrics(
    graph: nx.Graph,
    resolution: float,
) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"communities": 0, "modularity": 0.0}

    pagerank = nx.pagerank(graph, weight="weight")

    distance_graph = graph.copy()
    for _, _, data in distance_graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        data["distance"] = 1.0 / max(w, 1e-9)
    betweenness = nx.betweenness_centrality(distance_graph, weight="distance")

    try:
        eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        eigen = nx.eigenvector_centrality(graph, max_iter=1000, weight="weight")

    if graph.number_of_edges() > 0:
        partition = community_louvain.best_partition(
            graph,
            weight="weight",
            resolution=resolution,
        )
        modularity = community_louvain.modularity(partition, graph, weight="weight")
    else:
        partition = {n: 0 for n in graph.nodes()}
        modularity = 0.0

    community_colors = build_community_color_map(partition)
    for node in graph.nodes():
        cid = int(partition.get(node, 0))
        r, g, b = community_colors[cid]
        graph.nodes[node]["Community"] = cid
        graph.nodes[node]["PageRank"] = float(pagerank.get(node, 0.0))
        graph.nodes[node]["Betweenness"] = float(betweenness.get(node, 0.0))
        graph.nodes[node]["Eigenvector"] = float(eigen.get(node, 0.0))
        graph.nodes[node]["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": 0.95},
            "size": float(graph.nodes[node].get("NodeSize", 10.0)),
        }

    for u, v, data in graph.edges(data=True):
        cu = int(partition.get(u, 0))
        cv = int(partition.get(v, 0))
        if cu == cv:
            r, g, b = community_colors[cu]
            alpha = 0.32
        else:
            r, g, b = (130, 130, 130)
            alpha = 0.16
        data["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": alpha},
            "thickness": float(0.8 + 7.0 * data.get("weight", 0.0)),
        }

    return {
        "communities": float(len(set(partition.values()))),
        "modularity": float(modularity),
    }


def average_path_length_lcc(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    if nx.is_connected(graph):
        return float(nx.average_shortest_path_length(graph))
    largest_cc = max(nx.connected_components(graph), key=len)
    sub = graph.subgraph(largest_cc).copy()
    if sub.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_shortest_path_length(sub))


def print_summary(graph: nx.Graph, stats: Dict[str, float]) -> None:
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    avg_path = average_path_length_lcc(graph)
    print("\nNetwork Summary")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print(f"Density: {density:.6f}")
    print(f"Average path length (LCC): {avg_path:.6f}")
    print(f"Communities: {int(stats['communities'])}")
    print(f"Modularity: {stats['modularity']:.4f}")

    rows = []
    for node, data in graph.nodes(data=True):
        rows.append(
            {
                "repository": node,
                "pagerank": data.get("PageRank", 0.0),
                "betweenness": data.get("Betweenness", 0.0),
                "eigenvector": data.get("Eigenvector", 0.0),
                "community": data.get("Community", 0),
                "domain": data.get("DomainGroup", "Unknown"),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        print("\nTop 10 by weighted betweenness:")
        print(df.sort_values("betweenness", ascending=False).head(10).to_string(index=False))


def compute_community_seeded_positions(
    graph: nx.Graph,
    community_attr: str = "Community",
    community_gap: float = 6000.0,
    local_scale: float = 900.0,
) -> Dict[str, Tuple[float, float]]:
    comm_nodes: Dict[int, List[str]] = {}
    for node, data in graph.nodes(data=True):
        cid = int(float(data.get(community_attr, 0)))
        comm_nodes.setdefault(cid, []).append(node)

    meta = nx.Graph()
    for cid, members in comm_nodes.items():
        meta.add_node(cid, size=len(members))
    for u, v, data in graph.edges(data=True):
        cu = int(float(graph.nodes[u].get(community_attr, 0)))
        cv = int(float(graph.nodes[v].get(community_attr, 0)))
        if cu == cv:
            continue
        weight = float(data.get("weight", 1.0))
        if meta.has_edge(cu, cv):
            meta[cu][cv]["weight"] += weight
        else:
            meta.add_edge(cu, cv, weight=weight)

    community_ids = sorted(
        comm_nodes,
        key=lambda cid: (-len(comm_nodes[cid]), cid),
    )
    if len(community_ids) <= 1:
        anchors = {cid: (0.0, 0.0) for cid in comm_nodes}
    else:
        # Force community anchors apart. A spring layout on the community
        # metagraph often collapses strong communities into one visual blob.
        n_comm = len(community_ids)
        radius = community_gap / (2.0 * math.sin(math.pi / n_comm))
        anchors = {}
        for idx, cid in enumerate(community_ids):
            angle = (2.0 * math.pi * idx / n_comm) - (math.pi / 2.0)
            anchors[cid] = (math.cos(angle) * radius, math.sin(angle) * radius)

    positions: Dict[str, Tuple[float, float]] = {}
    for cid, members in comm_nodes.items():
        members_sorted = sorted(members)
        ax, ay = anchors.get(cid, (0.0, 0.0))
        if len(members_sorted) == 1:
            positions[members_sorted[0]] = (ax, ay)
            continue
        sub = graph.subgraph(members_sorted).copy()
        local = nx.spring_layout(
            sub,
            seed=42,
            weight="weight",
            k=1.7 / math.sqrt(max(2, sub.number_of_nodes())),
            iterations=200,
        )
        scale = local_scale * (1.0 + 0.12 * math.log10(len(members_sorted) + 1))
        for node in members_sorted:
            px, py = local[node]
            positions[node] = (ax + float(px) * scale, ay + float(py) * scale)
    return positions


def infer_reference_community(
    graph: nx.Graph,
    node: str,
    reference_graph: nx.Graph,
) -> int:
    scores: Counter[int] = Counter()
    for nbr in graph.neighbors(node):
        if nbr not in reference_graph:
            continue
        cid = int(float(reference_graph.nodes[nbr].get("Community", 0)))
        scores[cid] += float(graph[node][nbr].get("weight", 1.0))
    if scores:
        return int(scores.most_common(1)[0][0])
    if reference_graph.number_of_nodes() == 0:
        return 0
    return 0


def stable_extra_position(
    node: str,
    anchors: Dict[int, Tuple[float, float]],
    local_scale: float,
) -> Tuple[int, Tuple[float, float]]:
    community_ids = sorted(anchors) or [0]
    digest = hashlib.sha256(node.encode("utf-8")).digest()
    cid = community_ids[int.from_bytes(digest[:4], "big") % len(community_ids)]
    angle_raw = int.from_bytes(digest[4:8], "big") / float(2**32)
    radius_raw = int.from_bytes(digest[8:12], "big") / float(2**32)
    angle = 2.0 * math.pi * angle_raw
    radius = local_scale * (0.22 + 0.55 * radius_raw)
    ax, ay = anchors.get(cid, (0.0, 0.0))
    return cid, (ax + math.cos(angle) * radius, ay + math.sin(angle) * radius)


def apply_positions_to_graph(
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
) -> None:
    for node, (x, y) in positions.items():
        if node not in graph:
            continue
        graph.nodes[node]["x"] = float(x)
        graph.nodes[node]["y"] = float(y)
        viz = graph.nodes[node].get("viz", {})
        viz["position"] = {"x": float(x), "y": float(y), "z": 0.0}
        graph.nodes[node]["viz"] = viz


def recolor_graph_by_community(graph: nx.Graph) -> None:
    partition = {node: int(float(data.get("Community", 0))) for node, data in graph.nodes(data=True)}
    colors = build_community_color_map(partition)
    for node, data in graph.nodes(data=True):
        cid = partition.get(node, 0)
        r, g, b = colors[cid]
        viz = data.get("viz", {})
        viz["color"] = {"r": r, "g": g, "b": b, "a": 0.95}
        viz["size"] = float(data.get("NodeSize", 10.0))
        data["viz"] = viz

    for u, v, data in graph.edges(data=True):
        cu = partition.get(u, 0)
        cv = partition.get(v, 0)
        if cu == cv:
            r, g, b = colors[cu]
            alpha = 0.32
        else:
            r, g, b = (130, 130, 130)
            alpha = 0.16
        data["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": alpha},
            "thickness": float(0.8 + 7.0 * data.get("weight", 0.0)),
        }


def apply_reference_layout(
    graph: nx.Graph,
    reference_path: str,
    community_gap: float,
    local_scale: float,
) -> None:
    reference_graph = nx.read_gexf(reference_path)
    reference_positions: Dict[str, Tuple[float, float]] = {}
    comm_positions: Dict[int, List[Tuple[float, float]]] = {}
    for node, data in reference_graph.nodes(data=True):
        if "x" not in data or "y" not in data:
            raise RuntimeError(
                f"Reference graph lacks x/y coordinates: {reference_path}. "
                "Run mixed graph first with --seed-community-layout."
            )
        x = float(data["x"])
        y = float(data["y"])
        reference_positions[node] = (x, y)
        cid = int(float(data.get("Community", 0)))
        comm_positions.setdefault(cid, []).append((x, y))

    comm_anchor = {
        cid: (
            sum(x for x, _ in coords) / len(coords),
            sum(y for _, y in coords) / len(coords),
        )
        for cid, coords in comm_positions.items()
    }

    positions: Dict[str, Tuple[float, float]] = {}
    for node in graph.nodes:
        if node in reference_positions:
            positions[node] = reference_positions[node]
            graph.nodes[node].setdefault("SignalCommunity", graph.nodes[node].get("Community", 0))
            graph.nodes[node]["Community"] = int(
                float(reference_graph.nodes[node].get("Community", graph.nodes[node].get("Community", 0)))
            )
            continue
        cid, pos = stable_extra_position(node, comm_anchor, local_scale)
        graph.nodes[node]["Community"] = cid
        positions[node] = pos

    apply_positions_to_graph(graph, positions)
    recolor_graph_by_community(graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build repository-level AI backbone network from GitHub data"
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing GitHub token (default: GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/repo_raw_data_fork.json",
        help="Path to save/load normalized repository data",
    )
    parser.add_argument(
        "--gexf-output",
        default="outputs/graphs/repo/repository_backbone_fork.gexf",
        help="Path to save GEXF output",
    )
    parser.add_argument(
        "--per-domain-limit",
        type=int,
        default=REPO_LIMIT_PER_DOMAIN,
        help="Number of repositories to fetch for each of CV and NLP topics",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing raw-output file instead of fetching from GitHub",
    )
    parser.add_argument(
        "--signal-mode",
        choices=("mixed", "readme", "fork"),
        default="mixed",
        help="Similarity signal to build: mixed, readme-only, or fork-only",
    )
    parser.add_argument(
        "--readme-similarity-version",
        choices=("v1", "v2"),
        default="v2",
        help="README similarity implementation to use (default: v2 sparse TF-IDF)",
    )
    parser.add_argument(
        "--pre-min-weight",
        type=float,
        default=0.03,
        help="Minimum mixed similarity to create an edge before backbone filter",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.16,
        help="Minimum edge weight to keep in backbone",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Keep top-k weighted edges per node in backbone",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=2,
        help="k for k-core cleanup (0 disables)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.1,
        help="Louvain resolution parameter",
    )
    parser.add_argument(
        "--auto-relax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically relax backbone constraints if graph is too small",
    )
    parser.add_argument(
        "--target-min-nodes",
        type=int,
        default=40,
        help="Target minimum nodes after backbone extraction in auto-relax mode",
    )
    parser.add_argument(
        "--weight-fork",
        type=float,
        default=0.6,
        help="Fork-overlap weight in mixed similarity",
    )
    parser.add_argument(
        "--weight-readme",
        type=float,
        default=0.2,
        help="README semantic similarity weight in mixed similarity",
    )
    parser.add_argument(
        "--weight-dep",
        type=float,
        default=0.2,
        help="Dependency overlap weight in mixed similarity",
    )
    parser.add_argument(
        "--seed-community-layout",
        action="store_true",
        help="Write community-clustered x/y coordinates based on this graph's Community attribute",
    )
    parser.add_argument(
        "--layout-reference-gexf",
        default="",
        help="Optional mixed/reference GEXF with Community and x/y positions to reuse",
    )
    parser.add_argument(
        "--community-gap",
        type=float,
        default=6000.0,
        help="Distance between community anchors for seeded layout",
    )
    parser.add_argument(
        "--local-scale",
        type=float,
        default=900.0,
        help="Scale for local node placement inside each community",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_exists = os.path.exists(args.raw_output)

    if args.reuse_raw:
        if not raw_exists:
            raise RuntimeError(
                f"--reuse-raw was set, but raw file does not exist: {args.raw_output}"
            )
        with open(args.raw_output, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        repo_data = normalize_loaded_dataset(raw_records)
        print(f"[load] reused raw data <- {args.raw_output} ({len(repo_data)} repos)")
    else:
        token = os.getenv(args.token_env, "").strip()
        if token:
            token = load_github_token(args.token_env)
            repo_data = build_dataset_from_github(
                token=token,
                per_domain_limit=args.per_domain_limit,
            )
            ensure_parent_dir(args.raw_output)
            with open(args.raw_output, "w", encoding="utf-8") as f:
                json.dump(repo_data, f, ensure_ascii=False, indent=2)
            print(f"[save] normalized raw repository data -> {args.raw_output}")
        elif raw_exists:
            with open(args.raw_output, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
            repo_data = normalize_loaded_dataset(raw_records)
            print(
                "[load] no GitHub token found; auto-reused raw data "
                f"<- {args.raw_output} ({len(repo_data)} repos)"
            )
        else:
            raise RuntimeError(
                f"Missing GitHub token in {args.token_env} and raw file not found: "
                f"{args.raw_output}. "
                "Set token to fetch from GitHub, or provide --reuse-raw with an existing raw file."
            )

    if not repo_data:
        raise RuntimeError("No repository data available after loading/fetching.")

    if args.signal_mode == "fork":
        weight_fork, weight_readme, weight_dep = 1.0, 0.0, 0.0
    elif args.signal_mode == "readme":
        weight_fork, weight_readme, weight_dep = 0.0, 1.0, 0.0
    else:
        weight_fork = args.weight_fork
        weight_readme = args.weight_readme
        weight_dep = args.weight_dep

    base_graph, signal_stats = build_repository_similarity_graph(
        repo_data=repo_data,
        pre_min_weight=args.pre_min_weight,
        weight_fork=weight_fork,
        weight_readme=weight_readme,
        weight_dep=weight_dep,
        readme_similarity_version=args.readme_similarity_version,
    )
    print(
        "[graph] pre-backbone "
        f"nodes={base_graph.number_of_nodes()} edges={base_graph.number_of_edges()}"
    )
    print(
        "[signal] "
        f"pair_count={int(signal_stats['pairs'])}, "
        f"fork_nonzero={signal_stats['fork_nonzero_ratio']:.3f}, "
        f"readme_nonzero={signal_stats['readme_nonzero_ratio']:.3f}, "
        f"dep_nonzero={signal_stats['dep_nonzero_ratio']:.3f}, "
        f"edge_keep_pre={signal_stats['edge_keep_ratio_pre']:.3f}"
    )
    print(
        "[weights] "
        f"mode={args.signal_mode}, "
        f"fork={signal_stats['weight_fork']:.3f}, "
        f"readme={signal_stats['weight_readme']:.3f}, "
        f"dep={signal_stats['weight_dep']:.3f}, "
        f"readme_version={signal_stats['readme_similarity_version']}"
    )

    if args.auto_relax:
        target_min_nodes = max(8, min(args.target_min_nodes, base_graph.number_of_nodes()))
        backbone, chosen = auto_relax_backbone(
            base_graph=base_graph,
            min_weight=args.min_weight,
            top_k=args.top_k,
            k_core_k=args.k_core,
            resolution=args.resolution,
            target_min_nodes=target_min_nodes,
        )
        print(
            "[backbone-config] "
            f"auto_relax=True, min_weight={chosen['min_weight']:.4f}, "
            f"top_k={int(chosen['top_k'])}, k_core={int(chosen['k_core'])}, "
            f"est_modularity={chosen['modularity']:.4f}"
        )
    else:
        backbone = apply_backbone_filter(
            graph=base_graph,
            min_weight=args.min_weight,
            top_k=args.top_k,
            k_core_k=args.k_core,
        )
        print(
            "[backbone-config] "
            f"auto_relax=False, min_weight={args.min_weight:.4f}, "
            f"top_k={args.top_k}, k_core={args.k_core}"
        )

    print(
        "[graph] backbone "
        f"nodes={backbone.number_of_nodes()} edges={backbone.number_of_edges()}"
    )

    stats = annotate_graph_metrics(backbone, resolution=args.resolution)
    for node in backbone.nodes:
        backbone.nodes[node]["SignalCommunity"] = int(float(backbone.nodes[node].get("Community", 0)))
    if args.layout_reference_gexf:
        apply_reference_layout(
            backbone,
            reference_path=args.layout_reference_gexf,
            community_gap=args.community_gap,
            local_scale=args.local_scale,
        )
        print(f"[layout] reused reference layout <- {args.layout_reference_gexf}")
    elif args.seed_community_layout:
        positions = compute_community_seeded_positions(
            backbone,
            community_attr="Community",
            community_gap=args.community_gap,
            local_scale=args.local_scale,
        )
        apply_positions_to_graph(backbone, positions)
        print("[layout] seeded community layout from this graph")

    ensure_parent_dir(args.gexf_output)
    nx.write_gexf(backbone, args.gexf_output)
    print(f"[save] gexf -> {args.gexf_output}")
    print_summary(backbone, stats)


if __name__ == "__main__":
    main()
