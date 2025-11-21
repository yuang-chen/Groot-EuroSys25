#!/usr/bin/env python3
"""
TCA Reordering using k-NN + MST/Graph Traversal

This approach:
1. Builds a k-NN graph where each row connects to k most similar neighbors using PyNNDescent
2. Optionally extracts a Minimum Spanning Tree (MST) from the k-NN graph
3. Traverses the graph/MST using DFS or BFS to produce final ordering

This creates a continuous path through the similarity space for optimal locality.

Optimizations applied:
- PyNNDescent for fast k-NN (O(N log N) with Numba)
- Vectorized edge reordering with NumPy (~100x faster)
- Dict comprehension for reorder mapping (cleaner, faster)
- k-NN neighbor graph caching for instant reuse
- CUSTOM SPARSE METRICS: Uses optimized sparse.py implementations

Dependencies:
- PyNNDescent: pip install pynndescent
- sparse.py (Must be in same directory)

Author: YuAng
Date: 2025-11-17
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from pynndescent import NNDescent
import time
import random
import os
import os.path as osp
import argparse
from collections import deque
import numba

# Import custom sparse metrics from sparse.py
from sparse import sparse_jaccard, sparse_hamming

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2022)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='YeastH', help="dataset")
parser.add_argument("--knn", type=int, default=16, 
                   help="number of nearest neighbors per row (default: 16)")
parser.add_argument("--similarity_metric", type=str, default='jaccard',
                   choices=['jaccard', 'hamming'],
                   help="similarity metric: jaccard (default) or hamming")
parser.add_argument("--no_mst", dest='use_mst', action='store_false', default=True,
                   help="disable MST extraction, use k-NN graph directly (MST enabled by default)")
parser.add_argument("--traversal", type=str, default='dfs',
                   choices=['dfs', 'bfs'],
                   help="graph traversal method: dfs (default) or bfs")
parser.add_argument("--start_node", type=str, default='max_degree',
                   choices=['max_degree', 'random', 'first'],
                   help="starting node selection: max_degree (default), random, or first")
parser.add_argument("--cache_dir", type=str, default="./knn_cache",
                   help="directory to cache k-NN indices (default: ./knn_cache)")
parser.add_argument("--no_cache", action='store_true',
                   help="disable k-NN index caching (rebuild every time)")
parser.add_argument("--force_rebuild", action='store_true',
                   help="force rebuild k-NN index even if cache exists")
parser.add_argument("--input_dir", type=str, default="./toydata",
                   help="directory containing input matrices (default: ./toydata)")
parser.add_argument("--output_dir", type=str, default="./toydata",
                   help="directory to save reordered matrices (default: ./toydata)")
args = parser.parse_args()
print(args)

## Load matrix from files - supports both .npz and binary CSR formats
def load_binary_csr(filepath, dtype=np.int32):
    """
    Load binary CSR format (same as C++ code):
    - IndexType nrow
    - IndexType nnz
    - IndexType row_ptr[nrow+1]
    - IndexType col_idx[nnz]
    """
    print(f"Loading binary CSR format from {filepath}")
    with open(filepath, 'rb') as f:
        # Read header
        nrow = np.fromfile(f, dtype=dtype, count=1)[0]
        nnz = np.fromfile(f, dtype=dtype, count=1)[0]
        
        print(f"  nrow={nrow}, nnz={nnz}")
        
        # Read row pointers
        row_ptr = np.fromfile(f, dtype=dtype, count=nrow + 1)
        
        # Read column indices
        col_idx = np.fromfile(f, dtype=dtype, count=nnz)
        
    print(f"  Binary CSR loaded successfully")
    return nrow, nnz, row_ptr, col_idx

def save_binary_csr(filepath, nrow, nnz, row_ptr, col_idx, dtype=np.int32):
    """
    Save binary CSR format (same as C++ code):
    - IndexType nrow
    - IndexType nnz
    - IndexType row_ptr[nrow+1]
    - IndexType col_idx[nnz]
    """
    print(f"Saving binary CSR format to {filepath}")
    with open(filepath, 'wb') as f:
        np.array([nrow], dtype=dtype).tofile(f)
        np.array([nnz], dtype=dtype).tofile(f)
        row_ptr.astype(dtype).tofile(f)
        col_idx.astype(dtype).tofile(f)
    print(f"  Binary CSR saved successfully")

dataset = args.dataset

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Try different file formats
npz_path = osp.join(args.input_dir, dataset + ".npz")
csr_path = osp.join(args.input_dir, dataset + ".csr")
bin_path = osp.join(args.input_dir, dataset + ".bin")

if osp.exists(npz_path):
    print(f"Loading from NPZ format: {npz_path}")
    matrix = np.load(npz_path)
    src_li = matrix['src_li']
    dst_li = matrix['dst_li']
    num_row = matrix['num_nodes']
    num_col = num_row
    num_nnz = len(src_li)
elif osp.exists(csr_path):
    print(f"Loading from binary CSR format: {csr_path}")
    # Try int32 first, then int64
    try:
        num_row, num_nnz, row_ptr, col_idx = load_binary_csr(csr_path, dtype=np.int32)
    except:
        print("  Retrying with int64...")
        num_row, num_nnz, row_ptr, col_idx = load_binary_csr(csr_path, dtype=np.int64)
    
    # Convert CSR to edge list (src_li, dst_li)
    src_li = []
    dst_li = []
    for i in range(num_row):
        for j in range(row_ptr[i], row_ptr[i+1]):
            src_li.append(i)
            dst_li.append(col_idx[j])
    src_li = np.array(src_li)
    dst_li = np.array(dst_li)
    num_col = num_row
elif osp.exists(bin_path):
    print(f"Loading from binary format: {bin_path}")
    try:
        num_row, num_nnz, row_ptr, col_idx = load_binary_csr(bin_path, dtype=np.int32)
    except:
        print("  Retrying with int64...")
        num_row, num_nnz, row_ptr, col_idx = load_binary_csr(bin_path, dtype=np.int64)
    
    # Convert CSR to edge list (src_li, dst_li)
    src_li = []
    dst_li = []
    for i in range(num_row):
        for j in range(row_ptr[i], row_ptr[i+1]):
            src_li.append(i)
            dst_li.append(col_idx[j])
    src_li = np.array(src_li)
    dst_li = np.array(dst_li)
    num_col = num_row
else:
    raise FileNotFoundError(f"Matrix file not found. Tried: {npz_path}, {csr_path}, {bin_path}")

print(f"Matrix: {num_row} x {num_col}, nnz: {num_nnz}")

scipy_coo = coo_matrix((np.ones(num_nnz, dtype=np.float32), (src_li, dst_li)), shape=(num_row, num_row))
scipy_csr = scipy_coo.tocsr()
row_ind = np.array(src_li)
col_ind = np.array(dst_li)

# Calculate row degrees for statistics
print("=== Calculating matrix statistics ===")
t_start = time.time()
row_degrees = scipy_csr.indptr[1:] - scipy_csr.indptr[:-1]  # Direct from CSR pointers
print(f"Matrix statistics calculated in {time.time() - t_start:.2f}s")
print(f"Row degree stats - min: {row_degrees.min()}, max: {row_degrees.max()}, "
      f"mean: {row_degrees.mean():.2f}, median: {np.median(row_degrees):.2f}")

print(f"Using similarity metric: {args.similarity_metric}")

# ============================================================================
# Prepare Custom Metric Functions
# ============================================================================

def create_hamming_metric(n_features):
    """
    Creates a closure for the sparse_hamming function to inject n_features,
    as PyNNDescent expects metric(ind1, data1, ind2, data2).
    """
    # We must use default argument val=n_features to capture the variable 
    # at definition time for the jitted function
    @numba.njit()
    def impl(ind1, data1, ind2, data2):
        return sparse_hamming(ind1, data1, ind2, data2, n_features)
    return impl

# Select the appropriate function from sparse.py
metric_func = None
metric_name_for_cache = ""

if args.similarity_metric == 'jaccard':
    metric_func = sparse_jaccard
    metric_name_for_cache = 'sparse_jaccard'
elif args.similarity_metric == 'hamming':
    # sparse_hamming needs the number of features (columns)
    metric_func = create_hamming_metric(num_col)
    metric_name_for_cache = 'sparse_hamming'
else:
    raise ValueError(f"Unknown metric: {args.similarity_metric}")

# ============================================================================
# Build k-NN Graph using Efficient Library
# ============================================================================

print(f"\n=== Building k-NN graph (k={args.knn}) ===")
t_knn_start = time.time()

# k-NN graph represented as adjacency list with weights
knn_graph = [[] for _ in range(num_row)]  # knn_graph[i] = [(neighbor_id, similarity), ...]

print(f"Building k-NN graph using PyNNDescent with custom metric: {metric_name_for_cache}...")

# Setup cache path
use_cache = not args.no_cache
cache_filename = None
index_loaded = False

if use_cache:
    os.makedirs(args.cache_dir, exist_ok=True)
    # Cache the neighbor graph (numpy arrays)
    cache_filename = osp.join(
        args.cache_dir, 
        f"{dataset}_k{args.knn}_{metric_name_for_cache}_neighbors.npz"
    )
    
    # Try to load cached neighbor graph
    if osp.exists(cache_filename) and not args.force_rebuild:
        print(f"  Loading cached neighbor graph from: {cache_filename}")
        try:
            t_load_start = time.time()
            cached_data = np.load(cache_filename, allow_pickle=True)
            neighbors = cached_data['neighbors']
            distances = cached_data['distances']
            print(f"  Neighbor graph loaded in {time.time() - t_load_start:.2f}s")
            index_loaded = True
        except Exception as e:
            print(f"  Warning: Failed to load cache ({e}), rebuilding...")
            index_loaded = False

# Build index if not loaded from cache
if not index_loaded:
    print(f"  Building index with custom metric, n_neighbors={args.knn + 1}")
    if args.force_rebuild:
        print("  (forced rebuild)")
    
    t_build_start = time.time()
    
    # Pass the actual function object to metric
    index = NNDescent(
        scipy_csr,
        metric=metric_func,
        n_neighbors=args.knn + 1,  # +1 to account for self
        random_state=2022,
        verbose=True,
        n_jobs=-1
    )
    print(f"  Index built in {time.time() - t_build_start:.2f}s")
    
    # Get neighbors and distances from the newly built index
    neighbors, distances = index.neighbor_graph
    
    # Save neighbor graph to cache
    if use_cache and cache_filename:
        print(f"  Saving neighbor graph to cache: {cache_filename}")
        try:
            np.savez_compressed(cache_filename, neighbors=neighbors, distances=distances)
            print(f"  Cache saved successfully")
        except Exception as e:
            print(f"  Warning: Failed to save cache ({e})")

print("Converting to k-NN graph with similarities...")
for i in range(num_row):
    for j, (neighbor_id, dist) in enumerate(zip(neighbors[i], distances[i])):
        if neighbor_id >= 0 and neighbor_id != i:  # Skip self
            # Convert distance to similarity
            # sparse.py metrics return distances (dissimilarity)
            # Similarity = 1.0 - distance
            similarity = 1.0 - dist if dist < 1.0 else 0.0
            knn_graph[i].append((neighbor_id, similarity))

t_knn_end = time.time()
print(f"k-NN graph built in {t_knn_end - t_knn_start:.2f}s")

# Calculate graph statistics
total_edges = sum(len(neighbors) for neighbors in knn_graph)
avg_neighbors = total_edges / num_row
print(f"k-NN graph statistics:")
print(f"  - Total directed edges: {total_edges}")
print(f"  - Average neighbors per node: {avg_neighbors:.2f}")

# ============================================================================
# Optional: Extract MST from k-NN graph
# ============================================================================

if args.use_mst:
    print("\n=== Extracting Minimum Spanning Tree ===")
    t_mst_start = time.time()
    
    # Convert k-NN graph to dense adjacency matrix for MST extraction
    from scipy.sparse import lil_matrix
    
    # Build symmetric distance matrix
    adj_matrix = lil_matrix((num_row, num_row), dtype=np.float32)
    for i in range(num_row):
        for neighbor_id, sim in knn_graph[i]:
            distance = 1.0 - sim  # Convert similarity to distance
            # Make symmetric by taking minimum distance
            if adj_matrix[i, neighbor_id] == 0 or distance < adj_matrix[i, neighbor_id]:
                adj_matrix[i, neighbor_id] = distance
            if adj_matrix[neighbor_id, i] == 0 or distance < adj_matrix[neighbor_id, i]:
                adj_matrix[neighbor_id, i] = distance
    
    adj_matrix_csr = adj_matrix.tocsr()
    
    # Extract MST
    print("  Computing MST...")
    mst = minimum_spanning_tree(adj_matrix_csr)
    mst_coo = mst.tocoo()
    
    # Convert MST to adjacency list
    mst_graph = [[] for _ in range(num_row)]
    for i, j, dist in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        sim = 1.0 - dist  # Convert back to similarity
        mst_graph[i].append((j, sim))
        mst_graph[j].append((i, sim))  # MST is undirected
    
    # Use MST for traversal
    graph_to_traverse = mst_graph
    
    t_mst_end = time.time()
    print(f"MST extracted in {t_mst_end - t_mst_start:.2f}s")
    print(f"MST statistics:")
    print(f"  - Total edges: {mst.nnz}")
    print(f"  - Average degree: {2 * mst.nnz / num_row:.2f}")
    
else:
    print("\n=== Skipping MST extraction, using k-NN graph directly ===")
    # Convert directed k-NN graph to undirected for traversal
    undirected_graph = [[] for _ in range(num_row)]
    for i in range(num_row):
        for neighbor_id, sim in knn_graph[i]:
            undirected_graph[i].append((neighbor_id, sim))
            # Add reverse edge if not already present
            reverse_exists = any(n == i for n, _ in knn_graph[neighbor_id])
            if not reverse_exists:
                undirected_graph[neighbor_id].append((i, sim))
    
    graph_to_traverse = undirected_graph

# ============================================================================
# Select Starting Node
# ============================================================================

print(f"\n=== Selecting starting node (strategy: {args.start_node}) ===")

if args.start_node == 'max_degree':
    # Start from node with highest degree in the graph
    degrees = [len(neighbors) for neighbors in graph_to_traverse]
    start_node = np.argmax(degrees)
    print(f"  Starting from node {start_node} (degree: {degrees[start_node]})")
elif args.start_node == 'random':
    start_node = random.randint(0, num_row - 1)
    print(f"  Starting from random node {start_node}")
else:  # 'first'
    start_node = 0
    print(f"  Starting from first node (0)")

# ============================================================================
# Graph Traversal (DFS or BFS)
# ============================================================================

print(f"\n=== Graph traversal using {args.traversal.upper()} ===")
t_traversal_start = time.time()

visited = set()
reorder = []

if args.traversal == 'dfs':
    print("Performing DFS traversal...")
    
    def dfs(node, graph, visited, reorder):
        visited.add(node)
        reorder.append(node)
        
        # Sort neighbors by similarity (descending) for consistent ordering
        neighbors = sorted(graph[node], key=lambda x: x[1], reverse=True)
        
        for neighbor_id, sim in neighbors:
            if neighbor_id not in visited:
                dfs(neighbor_id, graph, visited, reorder)
    
    # Start DFS from start_node
    dfs(start_node, graph_to_traverse, visited, reorder)
    
    # Handle disconnected components
    for node in range(num_row):
        if node not in visited:
            dfs(node, graph_to_traverse, visited, reorder)

elif args.traversal == 'bfs':
    print("Performing BFS traversal...")
    
    # BFS using queue
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        node = queue.popleft()
        reorder.append(node)
        
        # Sort neighbors by similarity (descending) for consistent ordering
        neighbors = sorted(graph_to_traverse[node], key=lambda x: x[1], reverse=True)
        
        for neighbor_id, sim in neighbors:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append(neighbor_id)
    
    # Handle disconnected components
    for node in range(num_row):
        if node not in visited:
            queue.append(node)
            visited.add(node)
            while queue:
                node = queue.popleft()
                reorder.append(node)
                neighbors = sorted(graph_to_traverse[node], key=lambda x: x[1], reverse=True)
                for neighbor_id, sim in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

t_traversal_end = time.time()
print(f"{args.traversal.upper()} traversal completed in {t_traversal_end - t_traversal_start:.2f}s")
print(f"Total nodes ordered: {len(reorder)}")

# Verify
assert len(reorder) == num_row, f"Reorder length mismatch: {len(reorder)} != {num_row}"
assert len(set(reorder)) == num_row, "Duplicate nodes in reorder"

# ============================================================================
# Save Results
# ============================================================================

print("\n=== Saving results ===")
t_save_start = time.time()

# Create output suffix
suffix_parts = ['knn', f'k{args.knn}']
if args.use_mst:
    suffix_parts.append('mst')
suffix_parts.append(args.similarity_metric)
suffix_parts.append(args.traversal)
suffix = ".".join(suffix_parts)

output_suffix = f".{suffix}"
print(f"Output suffix: {output_suffix}")

np.savez(osp.join(args.output_dir, dataset + output_suffix + ".reorder_id.npz"),
         reorder_id=reorder)

# Create mapping from old to new indices (optimized: dict comprehension)
d = {reorder[i]: i for i in range(len(reorder))}

# Reorder edges (optimized: vectorized with NumPy)
reorder_array = np.zeros(num_row, dtype=np.int32)
for old_id, new_id in d.items():
    reorder_array[old_id] = new_id

new_row_ind = reorder_array[row_ind].tolist()
new_col_ind = reorder_array[col_ind].tolist()

np.savez(osp.join(args.output_dir, dataset + output_suffix + ".reorder.npz"),
         src_li=new_row_ind, dst_li=new_col_ind, num_nodes=num_row)

# Also save in binary CSR format (for C++ compatibility)
print("Converting reordered matrix to CSR format...")
reordered_coo = coo_matrix((np.ones(num_nnz, dtype=np.float32), 
                           (new_row_ind, new_col_ind)), 
                           shape=(num_row, num_col))
reordered_csr = reordered_coo.tocsr()

csr_output_path = osp.join(args.output_dir, dataset + output_suffix + ".reorder.csr")
save_binary_csr(csr_output_path, num_row, num_nnz, 
                reordered_csr.indptr, reordered_csr.indices, dtype=np.int32)

t_save_end = time.time()
print(f"Save time: {t_save_end - t_save_start:.2f}s")