# Project Structure

## Root

- `README.md`: quick start and pipeline entry points
- `scripts/`: Python analysis pipelines
- `data/`: raw data and API caches
- `outputs/`: generated graph artifacts
- `reports/`: report source files
- `docs/`: documentation

## scripts/

- `build_repository_backbone.py`: GitHub GraphQL retrieval + repository graph + backbone + centrality
- `prepare_visual_backbone.py`: visualization-oriented sparsification and metagraph export
- `seed_community_layout.py`: injects seeded coordinates for community-separated Gephi layouts
- `analyze_company_communities.py`: company-level aggregation and community analysis

## data/

- `raw/`: normalized raw API payloads and owner/company cache

## outputs/

- `graphs/repo/`: repository network and visualization variants (`repo_viz_*`)
- `graphs/company/`: company network and visualization variants (`company_viz_*`)

## reports/

- `final_report.tex`: current report draft source
