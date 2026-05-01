#pragma once

#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../base_index.hpp"

#include <boost/geometry/index/parameters.hpp>
#include <boost/geometry/index/predicates.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <cstddef>
#include <iterator>
#include <chrono>
#include <malloc.h>

#include "phtree/phtree.h"
#include "phtree/phtree_multimap.h"

using namespace improbable::phtree;
//using namespace improbable::PhTreeMultiMap;

namespace bench { 
    namespace index {

    template<size_t dim>
    class PH_TREE : 
        public BaseIndex {

            using Point = point_t<dim>;
            using Box = box_t<dim>;
            using Points = std::vector<point_t<dim>>;
            using ValueT = size_t;

            // -----------------------------
            // Euclidean distance functor
            // -----------------------------
            struct EuclideanDistance {
                double operator()(const Point& a, const Point& b) const {
                    double d = 0;
                    for (size_t i = 0; i < dim; ++i)
                        d += (a[i] - b[i]) * (a[i] - b[i]);
                    return d;
                }
            };
            
            
            public:
                inline PH_TREE(Points& points) {

                    this->num_of_points = points.size();
                    size_t mem_before = get_mem_usage();

                    auto start = std::chrono::steady_clock::now();

                    // construct ph-tree
                    //code here
                    for (int i = 0 ; i < points.size(); i ++){

                        tree.emplace(points[i],idx);
                        idx++;
                    }

                    auto end = std::chrono::steady_clock::now();
                    size_t mem_after = get_mem_usage();
                  
                    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
                    // std::cout << "Index Size: " << (mem_after - mem_before) - this->num_of_points * sizeof(size_t) << " bytes\n";
                }

                ~PH_TREE() {
                    //code here
                    std::cout<< "PHtree deleted"<<std::endl;
                }


                inline Points range_query(Box& box) {

                    PhBoxD<dim> box_ph(box.min_corner(), box.max_corner());

                    auto start = std::chrono::steady_clock::now();
                    Points return_values;
                    
                    //code here
                    
                    tree.for_each(box_ph, [&](const auto& key, ValueT& value) {
                        return_values.emplace_back(key);
                    });
                    
                    auto end = std::chrono::steady_clock::now();
                    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    range_count++;

                    return return_values;
                }


                inline Points knn_query(Point& q, unsigned int k) {
                    auto start = std::chrono::steady_clock::now();
                    Points return_values;
                    std::vector<ValueT> results;

                    // Convert query point to PH-Tree point
                    PhPointD<dim> q_ph(q);

                    // Create kNN iterator
                    auto it = tree.begin_knn_query(k, q_ph, EuclideanDistance());
                    auto end_it = tree.end();

                    while (it != end_it) {
                        return_values.push_back(it.first());
                        ++it;
                    }

                    auto end = std::chrono::steady_clock::now();
                    knn_count++;
                    knn_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                    return return_values;
                }

                inline void insert(Point& q)
                {
                    auto start = std::chrono::steady_clock::now();
                    tree.insert(q, idx);
                    idx++;
                    this->num_of_points++;
                    auto end = std::chrono::steady_clock::now();
                    updates_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    
                }

                inline void remove(Point& q)
                {
                    auto start = std::chrono::steady_clock::now();
                    auto it = tree.lower_bound(q);
                    tree.erase(it);
                    num_of_points--;
                    auto end = std::chrono::steady_clock::now();
                    removes_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();    
                }

                inline size_t count() {
                    //code here
                    return  this->num_of_points;
                }

                inline size_t get_mem_usage() {
                    struct mallinfo2 mi = mallinfo2();
                    return mi.uordblks; // total bytes currently allocated
                }

            private:
                //PhTreeD<dim, ValueT> tree;
                PhTreeMultiMapD<dim, ValueT> tree; 
                size_t num_of_points;
                size_t idx = 0;
        };
    }
}