"""
Network-robustness target — viability pilot
Perth Public Transport Network Analysis — merged_node_variant

PROPOSAL UNDER TEST
    Instead of asserting index weights, fit them against a target that is
    (a) not circular with the components and (b) about robustness, which is what
    the thesis is about. The candidate target is the Latora-Marchiori efficiency
    loss caused by removing a stop:

        E(G)  = 1/(N(N-1)) * SUM_{i != j} 1 / d(i,j)
        dE(v) = ( E(G) - E(G - v) ) / E(G)

    Latora and Marchiori [18] applied precisely this to the Boston MBTA, so the
    method is established for transit networks and is already cited in Chapter 2.

TWO RISKS THIS PILOT EXISTS TO MEASURE
    1. COST. Exact E(G) needs all-pairs shortest paths: ~11,773 Dijkstras at
       ~50 ms each is ~10 minutes for ONE evaluation. Doing that per removed node
       is ~82 days. Infeasible as stated. Mitigation tested here: estimate E from
       a fixed set of k pivot sources, reused identically across the baseline and
       every removal so that differences are signal rather than sampling noise.

    2. CIRCULARITY. Efficiency loss is expected to correlate with betweenness,
       because both count paths routed through a node. If dE is essentially a
       restatement of C, fitting weights against it would recover C and the
       exercise would be empty. The pilot measures that correlation directly.

       The reason it may NOT be circular: betweenness counts paths through v but
       ignores the quality of the alternatives. A node carrying many paths that
       has a parallel route beside it causes little efficiency loss. So dE is
       closer to "betweenness x absence of redundancy" — which is much nearer to
       reliance than centrality alone, and is exactly the distinction the Hub
       Reliance Index is trying to draw.

DISTANCE METRIC
    Hop count (unweighted). Topological vulnerability studies conventionally use
    it, and it is ~10x cheaper than weighted Dijkstra. A weighted variant is a
    one-line change (`weight="weight"`) if travel cost is preferred; the pilot
    reports whether the two rank nodes differently.

Usage:
    python robustness_target_pilot.py --pivots 100 --sample 60
    python robustness_target_pilot.py --pivots 300 --sample 800   # full run
"""

import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")


def main_graph():
    with open(os.path.join(OUT, "graph_full.pkl"), "rb") as f:
        G = pickle.load(f)
    ug = G.to_undirected()
    lcc = max(nx.connected_components(ug), key=len)
    Gl = G.subgraph(lcc).copy()
    ns = [n for n, d in Gl.nodes(data=True) if d.get("stoptype") != "School"]
    Gm = Gl.subgraph(ns).copy()
    c = max(nx.connected_components(Gm.to_undirected()), key=len)
    return Gm.subgraph(c).copy().to_undirected()


def efficiency(G, pivots, weight=None):
    """Mean inverse shortest-path length, estimated from a fixed pivot set."""
    tot, cnt = 0.0, 0
    for s in pivots:
        if s not in G:
            continue
        if weight:
            d = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        else:
            d = nx.single_source_shortest_path_length(G, s)
        for t, dist in d.items():
            if t != s and dist > 0:
                tot += 1.0 / dist
                cnt += 1
    return tot / cnt if cnt else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pivots", type=int, default=100)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--weighted", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    G = main_graph()
    N = G.number_of_nodes()
    print(f"main analysis graph: {N:,} nodes / {G.number_of_edges():,} edges")

    cen = pd.read_csv(os.path.join(OUT, "centrality_results.csv"), low_memory=False)
    cen = cen.drop_duplicates("stopnumber").set_index("stopnumber")

    nodes = np.array([n for n in G.nodes()])
    pivots = list(rng.choice(nodes, size=min(args.pivots, N), replace=False))

    w = "weight" if args.weighted else None
    t0 = time.time()
    E0 = efficiency(G, pivots, w)
    t_one = time.time() - t0
    print(f"baseline efficiency {E0:.6f}  ({t_one:.2f}s per evaluation, "
          f"{args.pivots} pivots)")
    print(f"  -> full sweep of all {N:,} nodes would take "
          f"{t_one * N / 3600:.1f} h")

    # Stratify the sample across the centrality range so the pilot covers both
    # structurally trivial and structurally critical nodes.
    infl = cen["influence_score"].reindex(nodes)
    order = np.argsort(-infl.fillna(0).to_numpy())
    strata = np.array_split(order, 6)
    per = max(1, args.sample // 6)
    pick = np.concatenate([rng.choice(s, size=min(per, len(s)), replace=False)
                           for s in strata])
    cand = nodes[pick]
    print(f"\nremoving {len(cand)} sampled nodes (stratified by influence) ...")

    rows = []
    for i, v in enumerate(cand, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(cand)}")
        H = G.copy()
        H.remove_node(v)
        Ev = efficiency(H, [p for p in pivots if p != v], w)
        rows.append({
            "stopnumber": int(v),
            "dE": (E0 - Ev) / E0 if E0 else np.nan,
            "influence_score": float(cen["influence_score"].get(v, np.nan)),
            "betweenness": float(cen["betweenness"].get(v, np.nan)),
            "pagerank": float(cen["pagerank"].get(v, np.nan)),
            "degree": G.degree(v),
        })
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "robustness_pilot.csv"), index=False)

    print("\n" + "=" * 64)
    print("IS THE TARGET CIRCULAR WITH CENTRALITY?")
    print("=" * 64)
    for c in ["influence_score", "betweenness", "pagerank", "degree"]:
        p = r["dE"].corr(r[c])
        s = r["dE"].corr(r[c], method="spearman")
        print(f"  dE vs {c:<16} Pearson {p:>7.3f}   Spearman {s:>7.3f}")
    print()
    print(f"  dE: median {r['dE'].median():.3e}  max {r['dE'].max():.3e}  "
          f"zero or negative: {int((r['dE'] <= 0).sum())}/{len(r)}")
    print()
    print("  READING IT")
    print("   |r| > 0.9  -> the target restates centrality; the exercise is circular")
    print("   |r| ~ 0.6-0.8 -> substantial shared variance but real residual signal")
    print("   |r| < 0.5  -> dE is measuring something centrality does not capture")


if __name__ == "__main__":
    main()
