# Project Structure

## Root

- `README.md`: quick start and pipeline entry points
- `scripts/`: Python analysis pipelines
- `data/`: raw data and snapshots
- `outputs/`: generated graph/table artifacts
- `reports/`: report source files
- `notebooks/`: exploratory notebooks
- `docs/`: documentation
- `lib/`: web visualization libraries/assets

## scripts/

- `build_repository_backbone.py`: GitHub GraphQL retrieval + repository graph + backbone + centrality
- `prepare_visual_backbone.py`: visualization-oriented sparsification and metagraph export
- `seed_community_layout.py`: injects seeded coordinates for community-separated Gephi layouts
- `analyze_company_communities.py`: company-level aggregation and community analysis
- `build_ai_backbone.py`: legacy topic-level pipeline
- `data_retrieve.py`: legacy retrieval helper
- `graph_construction.py`: legacy graph construction helper

## data/

- `raw/`: normalized raw API payloads and owner/company cache
- `snapshots/`: historical JSON snapshots kept for reproducibility

## outputs/

- `graphs/repo/`: repository network and visualization variants (`repo_viz_*`)
- `graphs/company/`: company network and visualization variants (`company_viz_*`)
- `tables/`: CSV outputs (community summaries, node tables)

## reports/

- `final_report.tex`: current report draft source

## notebooks/

- `example.ipynb`: exploratory notebook

