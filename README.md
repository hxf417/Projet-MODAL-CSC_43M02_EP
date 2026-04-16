# GitHub Repositories Analysis for AI Topics

## Overview

This project analyzes highly starred GitHub repositories related to artificial intelligence, with an emphasis on the `computer-vision` and `nlp` ecosystems. The main objective is to build a topic-level network from repository metadata, detect communities of related technologies, and study how pure or mixed those communities are.

The current pipeline:

1. queries GitHub GraphQL for top repositories in AI-related domains
2. enriches each repository with topics, contributors, stars, forks, and language data
3. builds a bipartite graph between elite users and repository topics
4. projects that bipartite graph into a topic-topic backbone network
5. detects topic communities with NetworkX
6. exports both visualization-ready graph data and tabular community results

The project is designed for two complementary goals:

- exploratory visualization in Gephi
- quantitative analysis of topic communities through CSV and JSON exports

## Main Research Idea

The central hypothesis is that AI technologies can be represented as a network of topics connected through shared contributors. If the same experienced contributors appear across repositories using different topics, those topics may belong to the same technical ecosystem or act as bridges between ecosystems.

This lets us study questions such as:

- which AI topics are the most central in popular repositories
- which topics connect computer vision and NLP ecosystems
- whether communities are semantically coherent
- whether a community is domain-pure or cross-domain

## Architecture

### 1. Data Collection Layer

The pipeline starts by retrieving repositories from the GitHub GraphQL API.

- search is performed on top-starred repositories for `topic:computer-vision` and `topic:nlp`
- for each repository, the script retrieves:
  - `nameWithOwner`
  - `stargazerCount`
  - `forkCount`
  - `primaryLanguage`
  - `repositoryTopics`
  - `mentionableUsers`

Main file:

- `build_ai_backbone.py`

Legacy file:

- `data_retrieve.py`

### 2. Data Cleaning and Structuring Layer

The retrieved repositories are normalized into a clean JSON dataset. At this stage, the code:

- merges repositories appearing in multiple source domains
- removes bot-like users such as `[bot]` accounts and `web-flow`
- converts topics and users into deduplicated lists
- stores source domain information for later domain-purity analysis

Main output:

- `raw_data.json`

### 3. Graph Construction Layer

The network construction is based on a bipartite model:

- one side contains users
- the other side contains repository topics

Only elite users are kept in the final graph construction. In this project, elite users are contributors connected to at least 2 repositories in the dataset. This reduces noise and helps focus on users who actually connect technologies across repositories.

The bipartite graph is then projected into a topic-topic graph:

- nodes are topics
- edges indicate that the same elite users are associated with both topics
- edge weight is the number of shared elite users

Weak ties are removed to keep the backbone clearer.

### 4. Community Detection Layer

The topic graph is partitioned into communities using NetworkX community detection.

- preferred method: NetworkX Louvain, when available
- fallback method: greedy modularity communities

Each topic node is enriched with:

- `community`
- `PageRank`
- `Betweenness`
- `Group` (`CV`, `NLP`, `Cross-domain`, `Unknown`)
- `Language`
- `Stars`
- `Forks`
- `RepoCount`
- `SemanticLabel`

### 5. Purity Analysis Layer

Each detected community is analyzed with two kinds of purity:

#### Domain Purity

Domain purity measures whether a community is mostly composed of:

- `CV` topics
- `NLP` topics
- `Cross-domain` topics

The exported indicators are:

- `dominant_group`
- `domain_purity`
- `domain_entropy`

High domain purity means that most topics in the community belong to the same domain family.

#### Semantic Purity

Semantic purity measures whether a community is coherent around a technical theme such as:

- `LLM / NLP`
- `Computer Vision`
- `Multimodal AI`
- `ML Frameworks`
- `Robotics / Autonomous Driving`
- `Information Extraction`

The exported indicators are:

- `dominant_semantic_label`
- `proposed_label`
- `semantic_purity`
- `semantic_entropy`

Semantic labels are currently heuristic and based on topic keywords. They are meant to support interpretation and reporting, not to serve as ground truth labels.

## Main Files

### Active Pipeline

- `build_ai_backbone.py`: main end-to-end script for data retrieval, graph construction, community detection, purity analysis, and exports
- `raw_data.json`: cleaned repository dataset
- `ai_backbone_network.gexf`: topic backbone network for Gephi
- `community_membership.csv`: topic-level community membership table
- `community_summary.csv`: summary table for each community
- `community_purity.csv`: dedicated purity metrics per community
- `communities.json`: full JSON export with community summaries and members

### Legacy / Exploratory Files

- `graph_construction.py`: older visualization-oriented script based on the previous raw data schema
- `topic_projection.html`: older HTML visualization output
- `data_retrieve.py`: earlier data collection script
- `example.ipynb`: notebook for exploratory work

## Data Flow

The current workflow can be summarized as:

`GitHub GraphQL API`
-> `repository retrieval`
-> `raw_data.json`
-> `elite-user/topic bipartite graph`
-> `topic-topic projected graph`
-> `NetworkX community detection`
-> `GEXF + CSV + JSON exports`

## How To Run

From the project directory:

```powershell
$env:GITHUB_TOKEN="your_github_token_here"
python .\build_ai_backbone.py
```

This generates:

- `raw_data.json`
- `ai_backbone_network.gexf`
- `community_membership.csv`
- `community_summary.csv`
- `community_purity.csv`
- `communities.json`

You can also specify custom output paths:

```powershell
python .\build_ai_backbone.py `
  --raw-output .\raw_data.json `
  --gexf-output .\ai_backbone_network.gexf `
  --community-membership-output .\community_membership.csv `
  --community-summary-output .\community_summary.csv `
  --community-purity-output .\community_purity.csv `
  --communities-json-output .\communities.json
```

## Visualization

The main visualization format is the Gephi-compatible GEXF graph:

- open `ai_backbone_network.gexf` in Gephi
- apply a layout such as `ForceAtlas 2`
- color nodes by `community`
- size nodes by `PageRank` or `Stars`

This allows visual exploration of:

- major AI topic communities
- bridges between communities
- central topics in the ecosystem

## Interpretation of Outputs

### `community_membership.csv`

Use this table to inspect each topic individually:

- which community it belongs to
- its semantic label
- its centrality
- its stars, forks, and degree

### `community_summary.csv`

Use this table to compare communities globally:

- size
- density
- total stars and forks
- dominant domain
- dominant semantic label
- top topics

### `community_purity.csv`

Use this table specifically for the purity analysis section of the report:

- domain purity
- semantic purity
- entropy-based heterogeneity
- proposed high-level community label

### `communities.json`

Use this file when you want the full nested structure of communities and members for notebooks, dashboards, or additional analysis.

## Current Limitations

- the dataset is currently limited to repositories returned by the `computer-vision` and `nlp` topic searches
- semantic labels are heuristic and depend on topic names
- mentionable users are a proxy for contributor structure, but they are not a perfect measure of collaboration
- community purity depends on the quality of GitHub topic tagging

## Possible Next Improvements

- extend the repository domains beyond `computer-vision` and `nlp`
- add time-based analysis such as stars per month
- compare multiple community detection algorithms
- create a browser-based interactive visualization in addition to Gephi
- refine semantic labeling rules for more precise subfields

## Authors / Context

This repository was developed as part of the MODAL CSC_43M02_EP project on GitHub repository analysis, focused on AI technology ecosystems.
