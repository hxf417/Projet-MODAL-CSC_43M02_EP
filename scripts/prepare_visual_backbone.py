#!/usr/bin/env python3
"""Prepare clearer community-centric GEXF views for Gephi.

This script applies three visualization-focused steps:
1) backbone sparsification on the original graph;
2) community metagraph aggregation (community as super-node);
3) strongest inter-community edge extraction.
"""

from __future__ import annotations

import argparse
import colorsys
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import community as community_louvain
import networkx as nx

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


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def build_color_map(ids: Iterable[int]) -> Dict[int, Tuple[int, int, int]]:
    ids_sorted = sorted(set(ids))
    out: Dict[int, Tuple[int, int, int]] = {}
    for idx, cid in enumerate(ids_sorted):
        if idx < len(BASE_COLOR_PALETTE):
            out[cid] = BASE_COLOR_PALETTE[idx]
        else:
            hue = (idx * 0.61803398875) % 1.0
            sat = 0.62
            val = 0.92
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
            out[cid] = (int(r_f * 255), int(g_f * 255), int(b_f * 255))
    return out


def edge_weight(data: Dict, weight_attr: str) -> float:
    return max(0.0, to_float(data.get(weight_attr, data.get("weight", 0.0)), 0.0))


def infer_community_attr(graph: nx.Graph, requested: Optional[str]) -> Optional[str]:
    if requested:
        if graph.number_of_nodes() == 0:
            return requested
        first_node = next(iter(graph.nodes()))
        if requested in graph.nodes[first_node]:
            return requested
        return None

    candidates = [
        "Community",
        "community",
        "CompanyCommunity",
        "company_community",
        "partition",
    ]
    if graph.number_of_nodes() == 0:
        return None
    first_node = next(iter(graph.nodes()))
    first_data = graph.nodes[first_node]
    for c in candidates:
        if c in first_data:
            return c
    return None


def normalize_partition(raw_partition: Dict[str, object]) -> Dict[str, int]:
    label_to_id: Dict[str, int] = {}
    out: Dict[str, int] = {}
    next_id = 0
    for node, raw_label in raw_partition.items():
        key = str(raw_label)
        if key not in label_to_id:
            label_to_id[key] = next_id
            next_id += 1
        out[node] = label_to_id[key]
    return out


def get_partition(
    graph: nx.Graph,
    community_attr: Optional[str],
    recompute: bool,
    resolution: float,
    weight_attr: str,
) -> Tuple[Dict[str, int], str, float]:
    if graph.number_of_nodes() == 0:
        return {}, "none", 0.0

    use_existing = (not recompute) and community_attr is not None
    if use_existing:
        raw = {n: graph.nodes[n].get(community_attr, -1) for n in graph.nodes()}
        partition = normalize_partition(raw)
        source = f"existing:{community_attr}"
    else:
        partition = community_louvain.best_partition(
            graph, weight=weight_attr, resolution=resolution
        )
        source = "louvain"

    modularity = 0.0
    if graph.number_of_edges() > 0 and partition:
        modularity = float(community_louvain.modularity(partition, graph, weight=weight_attr))
    return partition, source, modularity


def apply_backbone(
    graph: nx.Graph,
    weight_attr: str,
    min_weight: float,
    edge_quantile: float,
    top_k: int,
    k_core_k: int,
    keep_lcc: bool,
) -> Tuple[nx.Graph, Dict[str, float]]:
    g = graph.copy()
    original_edges = g.number_of_edges()
    original_nodes = g.number_of_nodes()

    for u, v, d in list(g.edges(data=True)):
        d["weight"] = edge_weight(d, weight_attr)

    g.remove_edges_from(
        [(u, v) for u, v, d in g.edges(data=True) if d.get("weight", 0.0) < min_weight]
    )

    if g.number_of_edges() > 0 and edge_quantile > 0:
        weights = [to_float(d.get("weight", 0.0), 0.0) for _, _, d in g.edges(data=True)]
        q_thr = quantile(weights, edge_quantile)
        g.remove_edges_from(
            [(u, v) for u, v, d in g.edges(data=True) if to_float(d.get("weight", 0.0), 0.0) < q_thr]
        )
    else:
        q_thr = 0.0

    if top_k > 0 and g.number_of_edges() > 0:
        keep: Set[Tuple[str, str]] = set()
        for node in g.nodes():
            scored: List[Tuple[float, Tuple[str, str]]] = []
            for nbr in g.neighbors(node):
                w = to_float(g[node][nbr].get("weight", 0.0), 0.0)
                scored.append((w, tuple(sorted((str(node), str(nbr))))))
            scored.sort(reverse=True)
            for _, edge_key in scored[:top_k]:
                keep.add(edge_key)
        drop = []
        for u, v in g.edges():
            edge_key = tuple(sorted((str(u), str(v))))
            if edge_key not in keep:
                drop.append((u, v))
        g.remove_edges_from(drop)

    g.remove_nodes_from(list(nx.isolates(g)))

    if k_core_k > 0 and g.number_of_nodes() > 0:
        if max(dict(g.degree()).values(), default=0) >= k_core_k:
            g = nx.k_core(g, k=k_core_k)

    if keep_lcc and g.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()

    stats = {
        "nodes_before": float(original_nodes),
        "edges_before": float(original_edges),
        "nodes_after": float(g.number_of_nodes()),
        "edges_after": float(g.number_of_edges()),
        "edge_quantile_threshold": float(q_thr),
    }
    return g, stats


def set_node_viz(graph: nx.Graph, partition: Dict[str, int]) -> Dict[int, Tuple[int, int, int]]:
    color_map = build_color_map(partition.values())
    for node in graph.nodes():
        cid = int(partition.get(node, 0))
        r, g, b = color_map[cid]
        stars = to_float(graph.nodes[node].get("Stars", 0.0), 0.0)
        size = graph.nodes[node].get("NodeSize")
        if size is None:
            size = 8.0 + 5.0 * math.log10(stars + 1.0)
        graph.nodes[node]["Community"] = cid
        graph.nodes[node]["NodeSize"] = float(size)
        viz = graph.nodes[node].get("viz", {})
        viz["color"] = {"r": r, "g": g, "b": b, "a": 0.95}
        viz["size"] = float(size)
        graph.nodes[node]["viz"] = viz
    return color_map


def pick_strong_cross_pairs(
    graph: nx.Graph,
    partition: Dict[str, int],
    cross_quantile: float,
    cross_top_k: int,
) -> Set[Tuple[int, int]]:
    pair_weight: Dict[Tuple[int, int], float] = defaultdict(float)
    for u, v, d in graph.edges(data=True):
        cu = int(partition.get(u, -1))
        cv = int(partition.get(v, -1))
        if cu < 0 or cv < 0 or cu == cv:
            continue
        key = tuple(sorted((cu, cv)))
        pair_weight[key] += to_float(d.get("weight", 0.0), 0.0)

    if not pair_weight:
        return set()

    weights = list(pair_weight.values())
    thr = quantile(weights, cross_quantile) if cross_quantile > 0 else min(weights)
    strong = {k for k, w in pair_weight.items() if w >= thr}

    if cross_top_k > 0:
        by_comm: Dict[int, List[Tuple[float, Tuple[int, int]]]] = defaultdict(list)
        for pair, w in pair_weight.items():
            c1, c2 = pair
            by_comm[c1].append((w, pair))
            by_comm[c2].append((w, pair))
        for cid, scored in by_comm.items():
            scored.sort(reverse=True)
            for _, pair in scored[:cross_top_k]:
                strong.add(pair)
    return strong


def build_clustered_repo_view(
    graph: nx.Graph,
    partition: Dict[str, int],
    color_map: Dict[int, Tuple[int, int, int]],
    strong_pairs: Set[Tuple[int, int]],
) -> nx.Graph:
    g = graph.copy()
    for u, v, d in list(g.edges(data=True)):
        cu = int(partition.get(u, -1))
        cv = int(partition.get(v, -1))
        w = to_float(d.get("weight", 0.0), 0.0)
        if cu == cv:
            r, gg, b = color_map.get(cu, (120, 120, 120))
            alpha = 0.32
        else:
            key = tuple(sorted((cu, cv)))
            if key not in strong_pairs:
                g.remove_edge(u, v)
                continue
            r, gg, b = (95, 95, 95)
            alpha = 0.55
        d["viz"] = {
            "color": {"r": r, "g": gg, "b": b, "a": alpha},
            "thickness": float(0.6 + 5.0 * w),
        }

    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def build_community_metagraph(
    graph: nx.Graph,
    partition: Dict[str, int],
    color_map: Dict[int, Tuple[int, int, int]],
) -> nx.Graph:
    comm_members: Dict[int, List[str]] = defaultdict(list)
    for node, cid in partition.items():
        comm_members[int(cid)].append(node)

    m = nx.Graph()
    for cid, members in comm_members.items():
        stars = sum(to_float(graph.nodes[n].get("Stars", 0.0), 0.0) for n in members)
        pr = [to_float(graph.nodes[n].get("PageRank", 0.0), 0.0) for n in members]
        bt = [to_float(graph.nodes[n].get("Betweenness", 0.0), 0.0) for n in members]
        ev = [to_float(graph.nodes[n].get("Eigenvector", 0.0), 0.0) for n in members]
        size = 20.0 + 4.0 * math.sqrt(max(1.0, len(members)))
        r, g, b = color_map.get(cid, (120, 120, 120))
        m.add_node(
            str(cid),
            Label=f"C{cid}",
            Community=int(cid),
            RepoCount=int(len(members)),
            TotalStars=float(stars),
            MeanPageRank=float(sum(pr) / len(pr) if pr else 0.0),
            MeanBetweenness=float(sum(bt) / len(bt) if bt else 0.0),
            MeanEigenvector=float(sum(ev) / len(ev) if ev else 0.0),
            Members="|".join(sorted(members)),
            viz={"color": {"r": r, "g": g, "b": b, "a": 0.95}, "size": float(size)},
        )

    for u, v, d in graph.edges(data=True):
        cu = int(partition.get(u, -1))
        cv = int(partition.get(v, -1))
        if cu < 0 or cv < 0 or cu == cv:
            continue
        a, b = sorted((cu, cv))
        w = to_float(d.get("weight", 0.0), 0.0)
        if m.has_edge(str(a), str(b)):
            m[str(a)][str(b)]["weight_sum"] += w
            m[str(a)][str(b)]["edge_count"] += 1
        else:
            m.add_edge(
                str(a),
                str(b),
                weight_sum=float(w),
                edge_count=1,
            )

    for u, v, d in m.edges(data=True):
        w_sum = to_float(d.get("weight_sum", 0.0), 0.0)
        e_cnt = max(1, int(d.get("edge_count", 1)))
        mean_w = w_sum / e_cnt
        d["weight"] = float(w_sum)
        d["mean_weight"] = float(mean_w)
        d["viz"] = {
            "color": {"r": 95, "g": 95, "b": 95, "a": 0.45},
            "thickness": float(1.2 + 3.5 * math.log10(w_sum + 1.0)),
        }
    return m


def keep_strong_meta_edges(
    metagraph: nx.Graph,
    edge_quantile: float,
    top_k: int,
) -> nx.Graph:
    g = metagraph.copy()
    if g.number_of_edges() == 0:
        return g

    weights = [to_float(d.get("weight", 0.0), 0.0) for _, _, d in g.edges(data=True)]
    thr = quantile(weights, edge_quantile) if edge_quantile > 0 else min(weights)
    g.remove_edges_from(
        [(u, v) for u, v, d in g.edges(data=True) if to_float(d.get("weight", 0.0), 0.0) < thr]
    )

    if top_k > 0 and g.number_of_edges() > 0:
        keep: Set[Tuple[str, str]] = set()
        for node in g.nodes():
            scored: List[Tuple[float, Tuple[str, str]]] = []
            for nbr in g.neighbors(node):
                w = to_float(g[node][nbr].get("weight", 0.0), 0.0)
                scored.append((w, tuple(sorted((str(node), str(nbr))))))
            scored.sort(reverse=True)
            for _, edge_key in scored[:top_k]:
                keep.add(edge_key)
        drop = []
        for u, v in g.edges():
            if tuple(sorted((str(u), str(v)))) not in keep:
                drop.append((u, v))
        g.remove_edges_from(drop)

    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def write_outputs(
    input_graph: nx.Graph,
    backbone: nx.Graph,
    repo_view: nx.Graph,
    metagraph: nx.Graph,
    strong_metagraph: nx.Graph,
    output_prefix: str,
) -> None:
    input_out = f"{output_prefix}_input_copy.gexf"
    backbone_out = f"{output_prefix}_backbone.gexf"
    repo_view_out = f"{output_prefix}_clustered_view.gexf"
    meta_out = f"{output_prefix}_community_metagraph.gexf"
    meta_strong_out = f"{output_prefix}_community_metagraph_strong.gexf"

    nx.write_gexf(input_graph, input_out)
    nx.write_gexf(backbone, backbone_out)
    nx.write_gexf(repo_view, repo_view_out)
    nx.write_gexf(metagraph, meta_out)
    nx.write_gexf(strong_metagraph, meta_strong_out)

    print(f"[save] input_copy -> {input_out}")
    print(f"[save] backbone -> {backbone_out}")
    print(f"[save] clustered_repo_view -> {repo_view_out}")
    print(f"[save] community_metagraph -> {meta_out}")
    print(f"[save] community_metagraph_strong -> {meta_strong_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clearer cluster-centric GEXF views for Gephi."
    )
    parser.add_argument("--input-gexf", required=True, help="Input graph GEXF path")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output file prefix, e.g. outputs/repo500",
    )
    parser.add_argument(
        "--weight-attr",
        default="weight",
        help="Edge weight attribute name",
    )
    parser.add_argument(
        "--community-attr",
        default="",
        help="Existing node community attribute to reuse; empty means auto-detect",
    )
    parser.add_argument(
        "--recompute-community",
        action="store_true",
        help="Recompute Louvain communities on backbone graph",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution when recomputing communities",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop edges below this weight before other filters",
    )
    parser.add_argument(
        "--edge-quantile",
        type=float,
        default=0.75,
        help="Global weight quantile filter in [0,1], e.g. 0.75 keeps top 25 percent",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Keep top-k edges per node after quantile filtering",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=2,
        help="Apply k-core cleanup after edge filters (0 disables)",
    )
    parser.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only largest connected component for clearer figure",
    )
    parser.add_argument(
        "--cross-quantile",
        type=float,
        default=0.75,
        help="Cross-community pair strength quantile for bridge preservation",
    )
    parser.add_argument(
        "--cross-top-k",
        type=int,
        default=2,
        help="Always keep top-k strongest cross-community links per community",
    )
    parser.add_argument(
        "--meta-quantile",
        type=float,
        default=0.6,
        help="Metagraph edge quantile filter in [0,1]",
    )
    parser.add_argument(
        "--meta-top-k",
        type=int,
        default=3,
        help="Metagraph top-k edges per node",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = nx.read_gexf(args.input_gexf)

    backbone, bb_stats = apply_backbone(
        graph=graph,
        weight_attr=args.weight_attr,
        min_weight=args.min_weight,
        edge_quantile=args.edge_quantile,
        top_k=args.top_k,
        k_core_k=args.k_core,
        keep_lcc=args.keep_largest_component,
    )

    community_attr = infer_community_attr(
        backbone, requested=args.community_attr.strip() or None
    )
    partition, part_source, modularity = get_partition(
        graph=backbone,
        community_attr=community_attr,
        recompute=args.recompute_community,
        resolution=args.resolution,
        weight_attr="weight",
    )
    color_map = set_node_viz(backbone, partition)

    strong_pairs = pick_strong_cross_pairs(
        graph=backbone,
        partition=partition,
        cross_quantile=args.cross_quantile,
        cross_top_k=args.cross_top_k,
    )
    repo_view = build_clustered_repo_view(
        graph=backbone,
        partition=partition,
        color_map=color_map,
        strong_pairs=strong_pairs,
    )
    metagraph = build_community_metagraph(
        graph=backbone,
        partition=partition,
        color_map=color_map,
    )
    strong_metagraph = keep_strong_meta_edges(
        metagraph=metagraph,
        edge_quantile=args.meta_quantile,
        top_k=args.meta_top_k,
    )

    write_outputs(
        input_graph=graph,
        backbone=backbone,
        repo_view=repo_view,
        metagraph=metagraph,
        strong_metagraph=strong_metagraph,
        output_prefix=args.output_prefix,
    )

    print("\nSummary")
    print(
        "[backbone] "
        f"nodes={int(bb_stats['nodes_after'])}/{int(bb_stats['nodes_before'])}, "
        f"edges={int(bb_stats['edges_after'])}/{int(bb_stats['edges_before'])}, "
        f"edge_q_thr={bb_stats['edge_quantile_threshold']:.6f}"
    )
    print(
        "[community] "
        f"source={part_source}, communities={len(set(partition.values()))}, "
        f"modularity={modularity:.4f}"
    )
    print(
        "[clustered-view] "
        f"nodes={repo_view.number_of_nodes()}, edges={repo_view.number_of_edges()}, "
        f"strong_cross_pairs={len(strong_pairs)}"
    )
    print(
        "[metagraph] "
        f"nodes={metagraph.number_of_nodes()}, edges={metagraph.number_of_edges()} | "
        f"strong_nodes={strong_metagraph.number_of_nodes()}, "
        f"strong_edges={strong_metagraph.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
