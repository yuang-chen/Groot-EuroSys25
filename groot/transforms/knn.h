#pragma once


// C++ Standard Library
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <stack>
#include <string_view>
#include <unordered_set>
#include <utility>

// Third-party Libraries
#include <kgraph.h>
#include <thrust/system/omp/execution_policy.h>

namespace groot {

template<typename T>
using AdjVector = thrust::host_vector<thrust::host_vector<T>>;

template<typename T>
using AdjOracle = kgraph::VectorOracle<AdjVector<T>, thrust::host_vector<T>>;

template<typename T>
using AdjHash = std::unordered_map<int, std::vector<T>>;

template<typename T>
struct Tree {
    T          num_nodes{0};
    AdjHash<T> adjs;
};

template<typename T, typename U>
using MinHeapPair =
    std::priority_queue<std::pair<T, U>, thrust::host_vector<std::pair<T, U>>, std::greater<std::pair<T, U>>>;



auto sparse_hamming_distance = [](const thrust::host_vector<int>& a, const thrust::host_vector<int>& b) {
    float distance = 0;
    int   i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            ++i;
            ++distance;
        }
        else if (a[i] > b[j]) {
            ++j;
            ++distance;
        }
        else {
            ++i;
            ++j;
        }
    }
    // Remaining elements in array 'a' add to distance
    distance += a.size() - i;

    // Remaining elements in array 'b' add to distance
    distance += b.size() - j;

    return distance;
};


template<typename CSR>
void convert_csr_to_adj(const CSR& mat, AdjVector<int>& adj)
{
    adj.resize(mat.num_rows);

    thrust::host_vector<int> rowptr_h = mat.row_pointers;

#pragma omp parallel for
    for (int i = 0; i < mat.num_rows; i++) {
        const auto row_begin  = rowptr_h[i];
        const auto row_end    = rowptr_h[i + 1];
        const auto row_length = row_end - row_begin;

        adj[i].resize(row_length);

        thrust::copy(mat.column_indices.begin() + row_begin, mat.column_indices.begin() + row_end, adj[i].begin());
    }
}

template<typename Params>
void set_index_params(Params& index_params, int K, int L)
{
    index_params.K          = K;
    index_params.L          = L;
    index_params.reverse    = 0;
    index_params.iterations = 15;
    // index_params.S = 10;
    // index_params.R = 0.002;
    // index_params.controls = 0.7;
    // index_params.recall = 1;
    // index_params.delta = 100;
    // index_params.seed = top_k;
}

template<typename Params>
void set_search_params(Params& search_params, int K)
{
    search_params.K = K;
    // search_params.M = 100;
    // search_params.P = 0.0;
    // search_params.S = 1000;
    // search_params.T = 1;
    // search_params.epsilon = 0.0;
    // search_params.init = 1000;
    // search_params.seed = 1;
}


template<typename CSR1, typename CSR2>
auto build_KNN_offline(const CSR1& mat, CSR2& knn)
{
    const auto     nrow = mat.num_rows;
    AdjVector<int> graph;

    convert_csr_to_adj(mat, graph);
    AdjOracle<int> oracle(graph, sparse_hamming_distance);

    kgraph::KGraph::IndexParams index_params;
    //! parameter tuning:
    //! https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
    unsigned i_k = std::min<unsigned>(nrow - 1, 200);
    unsigned i_l = std::min<unsigned>(i_k + 50, 300);
    set_index_params(index_params, i_k, i_l);

    kgraph::KGraph* index = kgraph::KGraph::create();
    index->build(oracle, index_params);

    const unsigned nnz = nrow * i_k;
    ASSERT(nnz < std::numeric_limits<unsigned>::max());
    knn.resize(nrow, nrow, nnz);
    thrust::sequence(knn.row_pointers.begin(), knn.row_pointers.end(), unsigned(0), i_k);

#pragma omp parallel for
    for (unsigned i = 0; i < nrow; i++) {
        auto row_begin = i * i_k;
        index->get_nn(i, knn.column_indices.data() + row_begin, knn.values.data() + row_begin, &i_k, &i_l);
    }
    delete index;
}

template<typename CSR, typename COO>
void clean_graph(const CSR& csr, COO& coo)
{
    using IndexType = typename CSR::index_type;
    using ValueType = typename CSR::value_type;

    thrust::host_vector<IndexType> row_indices(csr.num_entries);

    // uA.num_rows = A.num_rows;
    // uA.num_cols = A.num_cols;
    // uA.num_entries= A.num_entries;
    thrust::host_vector<ValueType> values_h         = csr.values;
    thrust::host_vector<IndexType> row_pointers_h   = csr.row_pointers;
    thrust::host_vector<IndexType> column_indices_h = csr.column_indices;

    coo.resize(csr.num_rows, csr.num_cols, csr.num_entries);
    coo.column_indices = csr.column_indices;
    coo.values         = csr.values;
    get_row_indices_from_pointers(csr.row_pointers, coo.row_indices);

    auto row_col_begin =
        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    // Sort by (row, col)
    sort_columns_per_row(coo.row_indices, coo.column_indices, coo.values);

    // print_zeros(coo, "after sort");
    //  Remove duplicates
    auto unique_end  = thrust::unique_by_key(row_col_begin, row_col_begin + coo.num_entries, coo.values.begin());
    auto unique_size = thrust::distance(row_col_begin, unique_end.first);
    coo.resize(coo.num_rows, coo.num_cols, unique_size);

    // print_zeros(coo, "after unique");
    //  Remove self-loop
    auto row_col_val_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin(), coo.values.begin()));
    auto row_col_val_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end(), coo.values.end()));

    // Resize the matrix
    auto new_end  = thrust::remove_if(row_col_val_begin, row_col_val_end, IsSelfLoop<IndexType, ValueType>());
    int  new_size = thrust::distance(row_col_val_begin, new_end);
    coo.resize(coo.num_rows, coo.num_cols, new_size);

    printf("unique size %d, new size %d\n", unique_size, new_size);
    // print_zeros(coo, "after self-loop");
    // Sort by values
    thrust::sort_by_key(
        coo.values.begin(),
        coo.values.end(),
        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())));
    // print_zeros(coo, "after sort by values");
}

// Tune K, check connectivity
// TODO: parallel MST - https://github.com/abarankab/parallel-boruvka
//? Reference:
// https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-using-stl-in-c/
template<typename COO, typename Tree, typename Vector>
auto build_MST(const COO& coo, Tree& tree, Vector& roots)
{
    using T = typename Vector::value_type;
    using F = typename COO::value_type;

    F          MST_weights = 0.0;
    const auto nrow        = coo.num_rows;
    const auto nnz         = coo.num_entries;

    thrust::host_vector<T> parents(nrow);
    thrust::host_vector<T> ranks(nrow);

    thrust::sequence(parents.begin(), parents.end(), 0);
    // find
    auto find = [&parents](int i) {
        while (parents[i] != i) {
            i = parents[i];
        }
        return i;
    };
    // union with rank
    auto unite_rank = [&find, &ranks, &parents](T i, T j) {
        i = find(i);
        j = find(j);
        if (ranks[i] > ranks[j]) {
            std::swap(i, j);
        }
        parents[i] = j;
        if (ranks[i] == ranks[j]) {
            ranks[j]++;
        }
    };

    // union without rank
    auto unite = [&find, &ranks, &parents](T i, T j) {
        i          = find(i);
        j          = find(j);
        parents[i] = j;
    };

    // O(ElogV)  parents[source] = root
    // tree.adjs[source] = parents[source];
    for (T i = 0; i < nnz; i++) {
        auto source = coo.row_indices[i];
        auto target = coo.column_indices[i];
        auto weight = coo.values[i];

        if (find(source) != find(target)) {
            MST_weights += weight;
            unite_rank(source, target);
            tree.adjs[source].push_back(target);
            tree.adjs[target].push_back(source);
        }
    }
    tree.num_nodes = nrow;

    // verify for MST: num_edges = num_nodes - 1
    {
        auto num_edges = 0;
        for (const auto& [node, adjs] : tree.adjs) {
            num_edges += adjs.size();
        }
        printf("total edges in tree: %d, expected edges: %d\n", num_edges, 2 * tree.num_nodes - 2);
    }
    // print_vec(parents, "parents ", 16);
    // print_vec(ranks, "ranks ", 16);
    // Collect root nodes (nodes where parents[i] == i)
    std::copy_if(thrust::counting_iterator<T>(0),
                 thrust::counting_iterator<T>(nrow),
                 std::back_inserter(roots),
                 [parents_ptr = parents.data()](T i) { return i == parents_ptr[i]; });
    printf("roots.size() = %d\n", roots.size());

    return MST_weights;
}

template<typename Tree, typename Vector>
auto perform_DFS(const Tree& tree, const Vector& roots, Vector& new_ids)
{
    using T = typename Vector::value_type;

    const auto                  num_nodes = tree.num_nodes;
    T                           max_depth = 0;
    std::stack<std::pair<T, T>> node_stack;  // (node, depth)
    std::vector<bool>           visited(num_nodes, false);
    std::vector<T>              ordered_nodes;

    ordered_nodes.reserve(num_nodes);

    // auto max_root = *std::max_element(par, roots.begin(),
    // roots.end()); fmt::println("max_root: {}", max_root);
    // print_vec(roots, "roots ", 16);
    // print_vec(tree.row_pointers, "row_pointers ", 16);
    // print_vec(tree.column_indices, "column_indices ", 16);

    for (const auto root : roots) {
        node_stack.push({root, 0});
    }

    while (!node_stack.empty()) {
        auto [curr, depth] = node_stack.top();
        node_stack.pop();

        if (visited[curr] == true) {
            continue;
        }
        visited[curr] = true;
        ordered_nodes.push_back(curr);
        max_depth = std::max(max_depth, depth);

        if (tree.adjs.find(curr) == tree.adjs.end()) {
            continue;
        }
        for (auto it = tree.adjs.at(curr).rbegin(); it != tree.adjs.at(curr).rend(); ++it) {
            if (visited[*it] == false) {
                node_stack.push({*it, depth + 1});
            }
        }
    }
    // std::cout << adj.size() << " " << ordered_nodes.size() <<
    // std::endl;
    ASSERT(num_nodes == ordered_nodes.size());

    new_ids.resize(num_nodes);

    thrust::scatter(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(num_nodes),
                    ordered_nodes.begin(),
                    new_ids.begin());

    // print_vec(ordered_nodes, "ordered_nodes ", 16);
    // print_vec(new_ids, "new_ids ", 16);

    return max_depth;
}


template<typename CSR, typename Vector>
auto groot(const CSR& mat, Vector& new_ids)
{
    CPUTimer timer;
    //+++++++++
    //++ KNN ++
    //+++++++++
    std::cout << "Step 1: KNN" << std::endl;
    // Kgraph requires unsigned index type
    CsrMatrix<unsigned, float, host_memory> knn;

    timer.start();

    build_KNN_offline(mat, knn);  // reverse edges -> undirected
    timer.stop();
    printf("[kGraph] time (ms): %f \n", timer.elapsed());

    ASSERT(knn.num_entries == knn.row_pointers.back() && knn.num_entries == knn.column_indices.size());

    //+++++++++
    //++ MST ++
    //+++++++++
    std::cout << "Step 2: MST" << std::endl;
    //? csr -> coo
    CooMatrix<unsigned, float, host_memory> uknn;
    clean_graph(knn, uknn);  // prepare for MST

    //? csr -> coo
    Tree<unsigned>           tree;
    thrust::host_vector<int> roots;
    timer.start();
    auto weights = build_MST(uknn, tree, roots);
    timer.stop();
    printf("[MST] time (ms): %f \n", timer.elapsed());
    printf("total weights of MST: %.2f\n", weights);

    //++++++++++++
    //++ DFS ++
    //++++++++++++
    std::cout << "Step 3: DFS" << std::endl;
    timer.start();
    auto depth = perform_DFS(tree, roots, new_ids);
    timer.stop();
    printf("[DFS] time (ms): %f \n", timer.elapsed());
    ASSERT(new_ids.size() == mat.num_rows);

    printf("Max Depth: %d\n", depth);
}

}  // namespace groot
