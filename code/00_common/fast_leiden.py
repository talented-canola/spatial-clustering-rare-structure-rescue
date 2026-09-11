"""Fast Leiden that reproduces spateo's `calculate_leiden_partition` exactly
(same leidenalg RBConfigurationVertexPartition, resolution_parameter, seed=888,
n_iterations=-1) but builds the igraph graph with a single vectorized add_edges
call instead of spateo's per-edge Python loop."""
import numpy as np
import igraph
import leidenalg


def build_graph(adj):
    rows, cols = adj.nonzero()
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    return G


def fast_leiden(adj, resolution):
    G = build_graph(adj)
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=888,
        n_iterations=-1,
    )
    return np.array(part.membership, dtype=int)


def fast_leiden_weighted(adj, resolution):
    """Weighted Leiden: adjacency is a weighted sparse matrix (edge weights = adj.data)."""
    rows, cols = adj.nonzero()
    data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=G.es["weight"],
        resolution_parameter=resolution,
        seed=888,
        n_iterations=-1,
    )
    return np.array(part.membership, dtype=int)
