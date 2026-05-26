# AI Repository Network Analysis

This repository studies AI GitHub ecosystems (CV + NLP) with repository-level similarity networks, backbone extraction, community detection, and company-level aggregation.

## Project Layout

- `scripts/`: executable Python pipelines
- `data/raw/`: raw and cached API data
- `outputs/graphs/repo/`: repository-level GEXF outputs
- `outputs/graphs/company/`: company-level GEXF outputs
- `reports/`: LaTeX report sources
- `docs/`: project documentation

Detailed file map: `docs/PROJECT_STRUCTURE.md`.

## Main Pipelines

1. Build repository graph backbone:

```bash
python scripts/build_repository_backbone.py \
  --raw-output data/raw/repo_raw_data_fork.json \
  --gexf-output outputs/graphs/repo/repository_backbone_fork.gexf
```

2. Prepare clearer Gephi views (backbone + metagraph):

```bash
python scripts/prepare_visual_backbone.py \
  --input-gexf outputs/graphs/repo/repository_backbone_fork.gexf \
  --output-prefix outputs/graphs/repo/repo_viz \
  --community-attr Community \
  --edge-quantile 0.35 --top-k 8 --k-core 2 \
  --cross-quantile 0.7 --cross-top-k 2 \
  --meta-quantile 0.5 --meta-top-k 3
```

3. Seed community coordinates for Gephi:

```bash
python scripts/seed_community_layout.py \
  --input-gexf outputs/graphs/repo/repo_viz_clustered_view.gexf \
  --output-gexf outputs/graphs/repo/repo_viz_clustered_seeded.gexf \
  --community-attr Community \
  --community-gap 7000 --local-scale 900
```

4. Aggregate to company network:

```bash
python scripts/analyze_company_communities.py \
  --repo-raw data/raw/repo_raw_data_fork.json \
  --repo-graph outputs/graphs/repo/repository_backbone_fork.gexf \
  --company-graph outputs/graphs/company/company_network.gexf \
  --drop-owners-without-company
```

5. (Recommended) Save outputs in a dataset-specific folder:

```bash
python scripts/analyze_company_communities.py \
  --repo-raw data/raw/repo_raw_data_fork.json \
  --repo-graph outputs/graphs/repo/repository_backbone_fork.gexf \
  --fetch-owner-company \
  --drop-owners-without-company \
  --dataset-tag top100_cv_nlp
```

This writes files under:
- `outputs/datasets/top100_cv_nlp/graphs/company/company_network.gexf`
- `outputs/datasets/top100_cv_nlp/tables/company_nodes.csv`
- `outputs/datasets/top100_cv_nlp/tables/company_communities.csv`
- `outputs/datasets/top100_cv_nlp/tables/owner_company_map.csv`
- `outputs/datasets/top100_cv_nlp/cache/owner_company_cache.json`

## Current Key Outputs

- Repository graph: `outputs/graphs/repo/repository_backbone_fork.gexf`
- Repo clustered view: `outputs/graphs/repo/repo_viz_clustered_seeded.gexf`
- Company graph: `outputs/graphs/company/company_network.gexf`
- Company metagraph: `outputs/graphs/company/company_viz_community_metagraph.gexf`
- Owner-company mapping: `outputs/tables/owner_company_map.csv` (or dataset-specific path)
