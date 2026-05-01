#pragma once

#include "type.hpp"
#include "datautils.hpp"
#include <array>
#include <bits/types/struct_rusage.h>
#include <boost/geometry/algorithms/detail/distance/interface.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <chrono>
#include <sys/time.h>

namespace bench { namespace common {

constexpr size_t ipow(size_t base, int exp, size_t result = 1) {
    return exp < 1 ? result : ipow(base*base, exp/2, (exp % 2) ? result*base : result);
}


constexpr size_t find_grid_K(size_t N, size_t dim) {
    size_t temp = 2;
    while (ipow(temp, dim) < (N / 1024)) {
        temp++;
    }
    return temp;
}


template<size_t dim>
inline bool is_in_box(point_t<dim>& p, box_t<dim>& box) {
    // for (size_t d=0; d<dim; ++d) {
    //     if ((p[d] > box.max_corner()[d]) || (p[d] < box.min_corner()[d])) {
    //         return false;
    //     }
    // }

    // return true;
    
    return boost::geometry::covered_by(p, box);
}


template<size_t dim>
inline double eu_dist_square(point_t<dim>& p1, point_t<dim>& p2) {
    double acc = 0;
    for (size_t i=0; i<dim; ++i) {
        auto temp = p1[i] - p2[i];
        acc += temp * temp;
    }
    return acc;
}


template<size_t dim, typename std::enable_if <dim >= 4>::type* = nullptr>
inline double eu_dist(point_t<dim>& p1, point_t<dim>& p2) {
    return std::sqrt(eu_dist_square(p1, p2));
}


template<size_t dim, typename std::enable_if <dim == 3>::type* = nullptr>
inline double eu_dist(point_t<dim>& p1, point_t<dim>& p2) {
    return std::hypot(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]);
}


template<size_t dim, typename std::enable_if <dim == 2>::type* = nullptr>
inline double eu_dist(point_t<dim>& p1, point_t<dim>& p2) {
    return std::hypot(p1[0]-p2[0], p1[1]-p2[1]);
}


template<size_t dim>
inline void print_point(point_t<dim>& p, bool is_endl=true) {
    std::cout << std::fixed;
    std::cout << "Point(";
    for (size_t i=0; i<dim-1; ++i) {
        std::cout << p[i] << ", ";
    }
    if (is_endl)
        std::cout << p[dim-1] << ")" << std::endl;
    else
        std::cout << p[dim-1] << ")";
}


template<size_t dim>
inline void print_box(box_t<dim>& box) {
    std::cout << "Box(";
    print_point(box.min_corner(), false);
    std::cout << ", ";
    print_point(box.max_corner(), false);
    std::cout << ")" <<std::endl;
}


template<size_t dim>
inline void print_points(vec_of_point_t<dim>& points) {
    for (auto p : points) {
        print_point(p);
    }
}


template<size_t dim>
inline void print_knn_result(point_t<dim>& q, vec_of_point_t<dim>& knn) {
    std::sort(knn.begin(), knn.end(), 
        [&q](point_t<dim>& p1, point_t<dim>& p2) -> bool { 
            return eu_dist_square(p1, q) > eu_dist_square(p2, q); });

    for (auto p : knn) {
        print_point(p);
        std::cout << "dist=" << std::sqrt(eu_dist_square(p, q)) << std::endl;
    }
}


size_t recordsInLeaves = 0;
size_t recordsInInternals = 0;

// Boost rtree visitor to compute rtree statistics
template <typename Value, typename Options, typename Box, typename Allocators>
struct statistics : public boost::geometry::index::detail::rtree::visitor<Value, typename Options::parameters_type, Box, Allocators, typename Options::node_tag, true>::type
{
    typedef typename boost::geometry::index::detail::rtree::internal_node<Value, typename Options::parameters_type, Box, Allocators, typename Options::node_tag>::type internal_node;
    typedef typename boost::geometry::index::detail::rtree::leaf<Value, typename Options::parameters_type, Box, Allocators, typename Options::node_tag>::type leaf;

    inline statistics()
        : level(0)
        , levels(1) // count root
        , nodes(0)
        , leaves(0)
        , values(0)
        , values_min(0)
        , values_max(0)
        , tree_size(0)
    {}

    inline void operator()(internal_node const& n)
    {
        typedef typename boost::geometry::index::detail::rtree::elements_type<internal_node>::type elements_type;
        elements_type const& elements = boost::geometry::index::detail::rtree::elements(n);
        
        ++nodes; // count node
        tree_size += sizeof(internal_node);

        recordsInInternals += elements.size();

        size_t const level_backup = level;
        ++level;
        
        levels += level++ > levels ? 1 : 0; // count level (root already counted)
                
        for (typename elements_type::const_iterator it = elements.begin();
            it != elements.end(); ++it)
        {
            boost::geometry::index::detail::rtree::apply_visitor(*this, *it->second);
        }
        
        level = level_backup;
    }

    inline void operator()(leaf const& n)
    {   
        typedef typename boost::geometry::index::detail::rtree::elements_type<leaf>::type elements_type;
        elements_type const& elements = boost::geometry::index::detail::rtree::elements(n);

        ++leaves; // count leaves
        tree_size += sizeof(leaf);
        
        std::size_t const v = elements.size();
        recordsInLeaves += v;
        // count values spread per node and total
        values_min = (std::min)(values_min == 0 ? v : values_min, v);
        values_max = (std::max)(values_max, v);
        values += v;
    }
    
    std::size_t level;
    std::size_t levels;
    std::size_t nodes;
    std::size_t leaves;
    std::size_t values;
    std::size_t values_min;
    std::size_t values_max;
    std::size_t tree_size;
    // std::size_t inLeaves;
    // std::size_t inIntenrals;
};

// apply the rtree visitor
template <typename Rtree> inline
size_t get_boost_rtree_statistics(Rtree const& tree) {
    typedef boost::geometry::index::detail::rtree::utilities::view<Rtree> RTV;
    RTV rtv(tree);

    statistics<
        typename RTV::value_type,
        typename RTV::options_type,
        typename RTV::box_type,
        typename RTV::allocators_type
    > stats_v;

    rtv.apply_visitor(stats_v);
    std::cout << "internals: " << recordsInInternals << " -- leaves: " << recordsInLeaves << std::endl;
    std::cout << "numInternals: " << stats_v.nodes << " -- numLeaves: " << stats_v.leaves << std::endl;
    return stats_v.tree_size;
}

}
}
