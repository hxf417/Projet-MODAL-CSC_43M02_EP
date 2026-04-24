#!/usr/bin/env python3
"""Seed community-separated coordinates into a GEXF graph for Gephi."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def pick_community_attr(graph: nx.Graph, requested: str) -> str:
    if requested:
        return requested
    if graph.number_of_nodes() == 0:
        return "Community"
    sample_node = next(iter(graph.nodes()))
    sample = graph.nodes[sample_node]
    for key in ("Community", "community", "CompanyCommunity", "company_community"):
        if key in sample:
            return key
    return "Community"


def compute_seeded_positions(
    graph: nx.Graph,
    community_attr: str,
    community_gap: float,
    local_scale: float,
) -> Dict[str, Tuple[float, float]]:
    comm_nodes: Dict[int, List[str]] = defaultdict(list)
    for n, d in graph.nodes(data=True):
        comm_nodes[to_int(d.get(community_attr, 0), 0)].append(n)

    # Build community metagraph for anchor layout.
    m = nx.Graph()
    for cid, members in comm_nodes.items():
        m.add_node(cid, size=len(members))
    for u, v, d in graph.edges(data=True):
        cu = to_int(graph.nodes[u].get(community_attr, 0), 0)
        cv = to_int(graph.nodes[v].get(community_attr, 0), 0)
        if cu == cv:
            continue
        w = float(d.get("weight", 1.0))
        if m.has_edge(cu, cv):
            m[cu][cv]["weight"] += w
        else:
            m.add_edge(cu, cv, weight=w)

    if m.number_of_nodes() <= 1:
        anchors = {cid: (0.0, 0.0) for cid in comm_nodes}
    else:
        anchors_raw = nx.spring_layout(m, weight="weight", seed=42, k=2.5 / math.sqrt(max(2, m.number_of_nodes())))
        anchors = {
            cid: (float(pos[0]) * community_gap, float(pos[1]) * community_gap)
            for cid, pos in anchors_raw.items()
        }

    out: Dict[str, Tuple[float, float]] = {}
    for cid, members in comm_nodes.items():
        if len(members) == 1:
            out[members[0]] = anchors.get(cid, (0.0, 0.0))
            continue
        sub = graph.subgraph(members).copy()
        sub_pos = nx.spring_layout(
            sub,
            weight="weight",
            seed=42,
            k=1.6 / math.sqrt(max(2, sub.number_of_nodes())),
        )
        ax, ay = anchors.get(cid, (0.0, 0.0))
        scale = local_scale * (1.0 + 0.1 * math.log10(len(members) + 1))
        for n in members:
            px, py = sub_pos[n]
            out[n] = (ax + float(px) * scale, ay + float(py) * scale)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed community layout into GEXF.")
    parser.add_argument("--input-gexf", required=True)
    parser.add_argument("--output-gexf", required=True)
    parser.add_argument("--community-attr", default="")
    parser.add_argument("--community-gap", type=float, default=5000.0)
    parser.add_argument("--local-scale", type=float, default=800.0)
    args = parser.parse_args()

    g = nx.read_gexf(args.input_gexf)
    community_attr = pick_community_attr(g, args.community_attr.strip())
    positions = compute_seeded_positions(
        g,
        community_attr=community_attr,
        community_gap=args.community_gap,
        local_scale=args.local_scale,
    )

    for n, (x, y) in positions.items():
        node = g.nodes[n]
        node["x"] = float(x)
        node["y"] = float(y)
        viz = node.get("viz", {})
        viz["position"] = {"x": float(x), "y": float(y), "z": 0.0}
        node["viz"] = viz

    nx.write_gexf(g, args.output_gexf)
    print(
        f"[save] seeded layout -> {args.output_gexf} "
        f"(nodes={g.number_of_nodes()}, edges={g.number_of_edges()}, community_attr={community_attr})"
    )


if __name__ == "__main__":
    main()
