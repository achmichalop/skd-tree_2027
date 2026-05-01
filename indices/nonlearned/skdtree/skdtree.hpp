#pragma once

#include "../../../utils/type.hpp"
#include "../../../utils/common.hpp"
#include "../../base_index.hpp"
#include "tree_core.hpp"
#include "evaluation.hpp"
#include "dimRanking.hpp"
#include "partitioning.hpp"
#include "updates.hpp"
#include <algorithm>
#include <array>
#include <boost/geometry/geometries/box.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <ostream>
#include <streambuf>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>
#include <chrono>

/*
 * SKDTREE
 * -------
 * This class provides a high-level wrapper for the skd-tree index, exposing a
 * unified interface for index construction, querying, and dynamic updates.
 *
 * The wrapper is templated on dimensionality and internally dispatches all
 * operations to the appropriate tree implementation based on the selected
 * internal node layout (N16, N32, or N64). The tree configuration is determined
 * at construction time and remains fixed throughout the lifetime of the index.
 *
 * SKDTREE supports range queries, k-nearest neighbor (kNN) queries, as well as
 * point insertions and deletions. Each operation is routed through specialized
 * wrapper functions corresponding to the active tree type.
 *
 */

namespace bench { 
    namespace index {
        template<size_t Dim>
        class SKDTREE : public BaseIndex
        {
            using Point = point_t<Dim>;
            using Points = std::vector<Point>;
            using Box = box_t<Dim>;
            size_t numOfPoints = 0;
            Points &points;

            public :
                SKDTREE(Points& points): points(points)
                {
                    numOfPoints = points.size();

                    auto start = std::chrono::steady_clock::now();
                    select_tree_type<Dim>(points.size());
                    buildIndex<Dim>[treeType](points);
                    auto end = std::chrono::steady_clock::now();
                    
                    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
                }

                inline Points range_query(Box& box) {

                    auto start = std::chrono::steady_clock::now();
                    Points results;
                    results = (Points) rangeQuery<Dim>[treeType](box);
                    auto end = std::chrono::steady_clock::now();

                    range_count++;
                    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                    return results;
                }

                inline Points knn_query(Point& q, unsigned int k) {
                    auto start = std::chrono::steady_clock::now();
                    Points results;
                    results = knnQuery<Dim>[treeType](q, k);
                    auto end = std::chrono::steady_clock::now();

                    knn_count++;
                    knn_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                    
                    return results;
                }

                inline void insert(Point& q)
                {
                    auto start = std::chrono::steady_clock::now();
                    insertPoint<Dim>[treeType](q);
                    auto end = std::chrono::steady_clock::now();
                    updates_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                }


                inline void remove(Point& q)
                {
                    auto start = std::chrono::steady_clock::now();
                    removePoint<Dim>[treeType](q);
                    auto end = std::chrono::steady_clock::now();
                    removes_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                }

                inline size_t count() {
                    return numOfPoints;
                }

                inline size_t index_size()
                {
                    size_t total = 0;

                    if (internalNodePool)
                    {
                        total += internalNodePool->get_used_nodes()
                            * internalNodePool->get_node_size();
                    }

                    if (leafNodePool)
                    {
                        total += leafNodePool->get_used_nodes()
                            * sizeof(LeafNode<Dim>);
                    }

                    if (leafNodePool)
                    {
                        for (void* block : leafNodePool->get_blocks())
                        {
                            char* base = static_cast<char*>(block);

                            for (size_t i = 0; i < leafNodePool->get_block_size_bytes() / sizeof(LeafNode<Dim>); ++i)
                            {
                                LeafNode<Dim>* leaf =
                                    reinterpret_cast<LeafNode<Dim>*>(base + i * sizeof(LeafNode<Dim>));

                                if (leaf->slotuse == 0) continue;

                                for (size_t d = 0; d < Dim; ++d)
                                {
                                    total += leaf->records[d].size() * sizeof(double);
                                }
                            }
                        }
                    }

                    return total;
                }                
        };

    }
}
