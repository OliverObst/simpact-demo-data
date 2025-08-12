#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:14:47 2023

@author: oliver

Build and persist an information-theoretic neighbourhood graph from sensor
soil-moisture time series, then render a wide, layered plot.

Behaviour
- If mi_full.csv exists (and RECOMPUTE_MI is False), reload MI and skip JVM.
- Otherwise, compute MI with JIDT Kraskov1, save:
  mi_full.csv, mi_norm.csv, mi_bundle.npz, mi_pvals.csv, mi_meta.json.
- Always build a sparse top-1-incoming graph, compute a layered layout
  with vertical spread, and save:
  mi_top1_edges.csv, mi_top1_graph.gpickle, mi_pos.csv,
  mi_neighbourhood.pdf/png 

Inputs
- sopivot+idx.csv (preferred) or sopa.csv
- infodynamics.jar in working dir

Notes
- Positions are saved so you can replot without recomputing the layout.
- Set RECOMPUTE_MI=True to force recomputation over the current data window.
"""

from __future__ import annotations
import os
import json
import datetime as dt
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Optional: only used if we need to compute MI
try:
    import jpype  # type: ignore
except Exception:
    jpype = None

# -------------------- config --------------------
RECOMPUTE_MI = False  # set True to force new MI even if mi_full.csv exists
RNG_SEED = 42         # layout jitter seed
FIGSIZE = (7.4, 3.8)  # wide aspect for paper
FORCE_RELAYOUT = True  # set True once to regenerate positions for the ellipse layout, then flip back to False
OUT_PREFIX = "mi_"    # file name prefix


# Edge selection policy: keep top-1 incoming per node only if it passes thresholds
REQUIRE_SIGNIFICANT = True   # if True, require p-value <= ALPHA (when p-values available)
ALPHA = 0.05                 # significance level for MI edges
MIN_MI_NORM = 0.0            # minimum normalised MI [0..1] required for an edge (0 keeps top-1 always)

# Data directory
DATADIR = Path("../data")

# Ensure ../figures exists 
FIGDIR = Path("../figures")
FIGDIR.mkdir(exist_ok=True)

# Inputs
sopa_file = DATADIR / Path("sopa.csv")
sopivot_file = DATADIR / Path("sopivot+idx.csv")

# Outputs
MI_FULL = DATADIR / Path("mi_full.csv")
MI_NORM = DATADIR / Path("mi_norm.csv")
MI_BUNDLE = DATADIR / Path("mi_bundle.npz")
MI_PVALS = DATADIR / Path("mi_pvals.csv")
MI_EDGES = DATADIR / Path("mi_top1_edges.csv")
MI_GRAPH = DATADIR / Path("mi_top1_graph.gpickle")
MI_POS = DATADIR / Path("mi_pos.csv")
MI_META = DATADIR / Path("mi_meta.json")
FIG_PDF = FIGDIR / Path("mi_neighbourhood.pdf")
FIG_PNG = FIGDIR / Path("mi_neighbourhood.png")

# -------------------- helpers --------------------
def _normalise_positive(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    A = np.array(A, dtype=float, copy=True)
    mask = A > 0
    if not mask.any():
        return np.zeros_like(A), 0.0, 1.0
    mn = float(A[mask].min())
    mx = float(A[mask].max())
    out = (A - mn) / (mx - mn + 1e-9)
    out[~mask] = 0.0
    return out, mn, mx


def _load_data_for_mi() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Return (data matrix, headers, cleaned sopivot DF)."""
    if sopivot_file.exists():
        sopivot = pd.read_csv(sopivot_file, header=0, na_values=["NA"]).apply(
            pd.to_numeric, errors="coerce"
        ).dropna(how="any")
        headers = sopivot.columns.tolist()
        data = sopivot.to_numpy(dtype=float)
        return data, headers, sopivot
    # Fallback: build pivot from sopa.csv
    sopa = pd.read_csv(sopa_file, sep=",", header=0, usecols=[1, 2, 3])
    streams = list(set(sopa["stream"]))
    processed = [x for x in streams if x.endswith(".processed")]
    headers = [
        x.replace("meshnet.bicentennialpark.SoilMonitor.", "").replace(
            "-SM-SOPA.VWC.processed", ""
        )
        for x in processed
    ]
    sopa = sopa[sopa["stream"].isin(processed)]
    repl = {old: new for old, new in zip(processed, headers)}
    sopa["datetime"] = pd.to_datetime(sopa["datetime"])  # type: ignore[index]
    sopa["hour"] = sopa["datetime"].dt.hour
    sopa["day"] = sopa["datetime"].dt.date
    sopivot = (
        pd.pivot_table(
            data=sopa,
            index=["day", "hour"],
            columns="stream",
            values="value",
            aggfunc="mean",
            fill_value=np.nan,
        ).rename(columns=repl)
    )
    sopivot.to_csv(sopivot_file, index=True, header=True)
    sopivot = sopivot.dropna(how="any")
    data = sopivot.to_numpy(dtype=float)
    return data, headers, sopivot


def _compute_mi_with_jidt(data: np.ndarray, headers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise MI and p-values with JIDT Kraskov1; return (mi_df, p_df)."""
    if jpype is None:
        raise RuntimeError("jpype is required to compute MI but is not available.")

    # we can't start this JVM crap twice. 
    if not jpype.isJVMStarted():
        jarLocation = "./infodynamics.jar"
        # Start the JVM (add the "-Xmx" option with say 1024M if 
        # you get crashes due to not enough memory space)
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", f"-Djava.class.path={jarLocation}")

    calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    calc = calcClass()

    N = data.shape[1]
    permutations = 500
    result = np.zeros((N, N), dtype=float)
    pvals = np.ones((N, N), dtype=float)

    print("Computing mutual information between nodes (Kraskov estimator).")
    print(" with JIDT info dynamics toolkit v1.6 https://github.com/jlizier/jidt")

    for s in range(N):
        src = jpype.JArray(jpype.JDouble, 1)(data[:, s].tolist())
        for d in range(N):
            if s == d:
                continue
            dst = jpype.JArray(jpype.JDouble, 1)(data[:, d].tolist())
            calc.initialise()
            calc.setObservations(src, dst)
            result[s, d] = float(calc.computeAverageLocalOfObservations())
            measDist = calc.computeSignificance(permutations)
            # Some JIDT builds expose .pValue, others getPValue()
            try:
                pval = float(measDist.pValue)
            except Exception:
                pval = float(measDist.getPValue())
            pvals[s, d] = pval
            print(
                f"MI: ({s}->{d}) = {result[s,d]:.4f} nats; p={pval:.5f}"
            )

    mi_df = pd.DataFrame(result, index=headers, columns=headers)
    p_df = pd.DataFrame(pvals, index=headers, columns=headers)
    return mi_df, p_df


def _build_top1_graph(mi_df: pd.DataFrame,
                      mi_norm_df: pd.DataFrame,
                      *,
                      p_df: pd.DataFrame | None = None,
                      alpha: float = 0.05,
                      min_norm: float = 0.0,
                      require_significant: bool = False) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Build sparse digraph with at most one incoming edge per node.
    Keep the strongest incoming edge only if it passes thresholds; otherwise no edge.
    """
    headers = mi_df.columns.tolist()
    edges = []
    for d in headers:
        # strongest incoming source for target d
        col = mi_df[d].drop(labels=[d], errors="ignore")
        if col.empty:
            continue
        s = col.idxmax()
        val = float(mi_df.at[s, d])
        if not np.isfinite(val) or val <= 0:
            continue
        norm = float(mi_norm_df.at[s, d]) if (s in mi_norm_df.index and d in mi_norm_df.columns) else 0.0
        pval = float(p_df.at[s, d]) if (p_df is not None and s in p_df.index and d in p_df.columns) else None
        keep = (norm >= min_norm)
        if require_significant:
            keep = keep and (pval is not None and pval <= alpha)
        if keep:
            edges.append({
                "source": s,
                "target": d,
                "mi": val,
                "mi_norm": norm,
                "p": pval if pval is not None else np.nan,
            })
    edges_df = pd.DataFrame(edges)
    G = nx.DiGraph()
    G.add_nodes_from(headers)
    for e in edges:
        G.add_edge(e["source"], e["target"], mi=e["mi"], w=e["mi_norm"], p=e["p"])
    return G, edges_df


def _layered_layout_with_spread(G: nx.DiGraph, seed: int = 42) -> tuple[dict[str, tuple[float,float]], dict[str, int]]:
    # roots: no incoming
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    Gr = G.reverse(copy=False)

    dist: dict[str, int] = {}
    q: deque[str] = deque()
    for r in roots:
        dist[r] = 0
        q.append(r)
    while q:
        x = q.popleft()
        for y in Gr.neighbors(x):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    max_layer = max(dist.values()) if dist else 0
    layer_attr = {n: dist.get(n, max_layer + 1) for n in G.nodes()}

    # group by layer and order within layers by total incident weight
    layers: dict[int, list[str]] = defaultdict(list)
    for n, l in layer_attr.items():
        layers[l].append(n)

    def node_weight(n: str) -> float:
        w = 0.0
        for _, _, d in G.in_edges(n, data=True):
            w += d.get("w", 0.0)
        for _, _, d in G.out_edges(n, data=True):
            w += d.get("w", 0.0)
        return -w  # heavier first

    for l in layers:
        layers[l].sort(key=node_weight)

    rng = np.random.default_rng(seed)
    pos: dict[str, tuple[float, float]] = {}
    L_sorted = sorted(layers.keys())
    for ix, l in enumerate(L_sorted):
        nodes = layers[l]
        m = len(nodes)
        ys = np.linspace(0.0, 1.0, m) if m > 1 else np.array([0.5])
        ys = (ys - 0.5) + rng.normal(0, 0.03, size=m)  # centre and jitter
        for y, n in zip(ys, nodes):
            pos[n] = (ix, float(y))

    # stretch
    for n, (x, y) in list(pos.items()):
        pos[n] = (x * 2.4, y * 1.0)

    return pos, layer_attr


def ellipse_by_layer(G: nx.DiGraph, width: float = 10.0, height: float = 3.4, jitter: float = 0.02, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Place nodes around a wide ellipse, ordered by (layer, name)."""
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes(), key=lambda n: (G.nodes[n].get("layer", 0), str(n)))
    m = max(len(nodes), 1)
    pos: dict[str, tuple[float, float]] = {}
    for i, n in enumerate(nodes):
        t = 2 * np.pi * (i / m)
        x = 0.5 * width * np.cos(t) + rng.normal(0, jitter)
        y = 0.5 * height * np.sin(t) + rng.normal(0, jitter * 0.5)
        pos[n] = (x, y)
    return pos


def _plot_graph(G: nx.DiGraph, pos: dict[str, tuple[float, float]], out_pdf: Path, out_png: Path) -> None:
    edge_widths = [0.6 + 3.2 * G[u][v].get("w", 0.0) for u, v in G.edges()]
    edge_alphas = [0.15 + 0.65 * G[u][v].get("w", 0.0) for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    EDGE_KW = dict(arrowstyle="-|>", arrowsize=12, connectionstyle="arc3,rad=0.06")

    # Draw edges individually so alpha/width are respected
    for (e, w, a) in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[e], width=w, alpha=a,
            edge_color="#ff8c00", arrows=True, **EDGE_KW
        )

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="0.75",
                           linewidths=0.5, edgecolors="white")
    nx.draw_networkx_labels(G, pos, font_size=8)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#ff8c00", lw=2.8, label="MI edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.75",
               markersize=7, label="Sensor"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, ncol=2)

    ax.set_axis_off()
    plt.tight_layout()

    # save figures
    try:
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
    except Exception:
        pass

    plt.close(fig)


def main() -> None:
    # Load data
    data, headers, _ = _load_data_for_mi()

    # Compute or reload MI
    if RECOMPUTE_MI or not MI_FULL.exists():
        if jpype is None:
            raise RuntimeError("jpype is required to compute MI but is not available.")
        mi_df, p_df = _compute_mi_with_jidt(data, headers)
        mi_df.to_csv(MI_FULL)
        p_df.to_csv(MI_PVALS)
        mi_norm_arr, mn, mx = _normalise_positive(mi_df.values)
        mi_norm_df = pd.DataFrame(mi_norm_arr, index=mi_df.index, columns=mi_df.columns)
        mi_norm_df.to_csv(MI_NORM)
        np.savez_compressed(MI_BUNDLE, mi=mi_df.values, headers=np.array(headers), min=mn, max=mx)
        print(f"Computed MI and saved to {MI_FULL}")
    else:
        mi_df = pd.read_csv(MI_FULL, index_col=0)
        headers = mi_df.columns.tolist()
        print(f"Reloaded MI from {MI_FULL}")
        p_df = pd.read_csv(MI_PVALS, index_col=0) if MI_PVALS.exists() else None
        if MI_NORM.exists():
            mi_norm_df = pd.read_csv(MI_NORM, index_col=0)
        else:
            mi_norm_arr, mn, mx = _normalise_positive(mi_df.values)
            mi_norm_df = pd.DataFrame(mi_norm_arr, index=mi_df.index, columns=mi_df.columns)
            mi_norm_df.to_csv(MI_NORM)

    # Build sparse top-1 graph
    G, edges_df = _build_top1_graph(
        mi_df, mi_norm_df, p_df=p_df,
        alpha=ALPHA, min_norm=MIN_MI_NORM,
        require_significant=REQUIRE_SIGNIFICANT,
    )

    # Persist edges
    edges_df.to_csv(MI_EDGES, index=False)

    # Persist graph (pickle fallback for older networkx)
    try:
        nx.write_gpickle(G, MI_GRAPH)
    except Exception:
        import pickle
        with open(MI_GRAPH, "wb") as f:
            pickle.dump(G, f)

    # Layer attributes and positions
    _, layer_attr = _layered_layout_with_spread(G, seed=RNG_SEED)
    nx.set_node_attributes(G, layer_attr, "layer")

    if MI_POS.exists() and not FORCE_RELAYOUT:
        pos_df = pd.read_csv(MI_POS)
        pos = {row["node"]: (float(row["x"]), float(row["y"])) for _, row in pos_df.iterrows()}
    else:
        pos = ellipse_by_layer(G, width=10.0, height=3.4, jitter=0.02, seed=RNG_SEED)
        pd.DataFrame({
            "node": list(pos.keys()),
            "x": [pos[n][0] for n in pos],
            "y": [pos[n][1] for n in pos],
        }).to_csv(MI_POS, index=False)

    # Plot
    _plot_graph(G, pos, FIG_PDF, FIG_PNG)

    # Meta
    meta = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "recompute": RECOMPUTE_MI,
        "alpha": ALPHA,
        "min_mi_norm": MIN_MI_NORM,
        "require_significant": REQUIRE_SIGNIFICANT,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "files": {
            "full": str(MI_FULL),
            "norm": str(MI_NORM),
            "pvals": str(MI_PVALS),
            "edges": str(MI_EDGES),
            "graph": str(MI_GRAPH),
            "pos": str(MI_POS),
            "pdf": str(FIG_PDF),
            "png": str(FIG_PNG),
        },
    }
    with open(MI_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. Saved figure to {FIG_PDF} + {FIG_PNG}.")


if __name__ == "__main__":
    main()
