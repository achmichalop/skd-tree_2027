#pragma once

#include "nanoflann.hpp"
#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../base_index.hpp"

#include <cstddef>
#include <vector>
#include <chrono>
#include <malloc.h>

#ifdef HEAP_PROFILE
#include <gperftools/heap-profiler.h>
#endif

namespace bench { namespace index {

// kdtree adapter using nanoflann
template <size_t Dim, size_t MaxSplit=128>
class KDTree : public BaseIndex {

using Point = point_t<Dim>;
using Box = box_t<Dim>;
using Points = std::vector<point_t<Dim>>;


// ===== This example shows how to use nanoflann with these types of containers:
// using my_vector_of_vectors_t = std::vector<std::vector<double> > ;
//
// The next one requires #include <Eigen/Dense>
// using my_vector_of_vectors_t = std::vector<Eigen::VectorXd> ;
// =============================================================================

/** A simple vector-of-vectors adaptor for nanoflann, without duplicating the
 * storage. The i'th vector represents a point in the state space.
 *
 *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality
 *      for the points in the data set, allowing more compiler optimizations.
 *  \tparam num_t The type of the point coordinates (typ. double or float).
 *  \tparam Distance The distance metric to use: nanoflann::metric_L1,
 *          nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
 *  \tparam IndexType The type for indices in the KD-tree index
 *         (typically, size_t of int)
 */
template <
    class VectorOfVectorsType, typename num_t = double, int DIM = -1,
    class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    using self_t =
        KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance>;
    using metric_t =
        typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t =
        nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>;

    /** The kd-tree index for the user to call its methods as usual with any
     * other FLANN index */
    index_t* index = nullptr;

    /// Constructor: takes a const ref to the vector of vectors object with the
    /// data points
    KDTreeVectorOfVectorsAdaptor(
        const size_t /* dimensionality */, const VectorOfVectorsType& mat,
        const int leaf_max_size = 10)
        : m_data(mat)
    {
        assert(mat.size() != 0 && mat[0].size() != 0);
        const size_t dims = mat[0].size();
        if (DIM > 0 && static_cast<int>(dims) != DIM)
            throw std::runtime_error(
                "Data set dimensionality does not match the 'DIM' template "
                "argument");
        index = new index_t(
            static_cast<int>(dims), *this /* adaptor */,
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }



    ~KDTreeVectorOfVectorsAdaptor() { delete index; }

    const VectorOfVectorsType& m_data;

    /** Query for the \a num_closest closest points to a given point
     *  (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.
     *
     * \note nChecks_IGNORED is ignored but kept for compatibility with
     * the original FLANN interface.
     */
    inline void query_knn( const num_t* query_point, const size_t num_closest, IndexType* out_indices, num_t* out_distances_sq) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParameters());
    }


    inline size_t query_range(const typename index_t::BoundingBox query_box, const size_t num_closest, IndexType* out_indices, num_t* out_distances_sq) const
    {
        nanoflann::KNNResultSet<num_t> resultSet(num_closest);       
        resultSet.init(out_indices, out_distances_sq);
        const auto nFound = index->findWithinBox(resultSet, query_box);
    

        // std::cout<<"nFound = " << nFound<<std::endl;
        // std::cout<<"result set size = " << resultSet.size()<<std::endl;
        //std::cout<<"result set capacity = " << resultSet.capacity<<std::endl;

        return nFound;
    }


    inline size_t query_range_v2(const typename index_t::BoundingBox query_box, std::vector<IndexType>& out_indices) const
    {
        nanoflann::RangeResultSet<IndexType> resultSet(out_indices);
        const auto nFound = index->findWithinBox_v2(resultSet, query_box);
        return nFound;
    }


    // inline size_t query_range(const typename index_t::BoundingBox& query_box, std::vector<IndexType>& out_indices) const
    // {
    //     out_indices.clear();
    //     return index->findWithinBox(out_indices, query_box);
    // }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t& derived() const { return *this; }
    self_t&       derived() { return *this; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return m_data[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    // Return true if the BBOX was already computed by the class and returned
    // in "bb" so it can be avoided to redo it again. Look at bb.size() to
    // find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

    /** @} */

};  // end of KDTreeVectorOfVectorsAdaptor


// customized kdtree type
using kdtree_t = KDTreeVectorOfVectorsAdaptor<Points, double, Dim>;

public:
KDTree(Points& points) {
    std::cout << "Construct kd-tree nanoflann MaxSplit=" << MaxSplit << std::endl;
    size_t mem_before = get_mem_usage();
    auto start = std::chrono::steady_clock::now();

#ifdef HEAP_PROFILE
    HeapProfilerStart("kdtree");
#endif

    kdtree = new kdtree_t(Dim, points, MaxSplit);
    kdtree->index->buildIndex();
    
#ifdef HEAP_PROFILE
    HeapProfilerDump("final");
    HeapProfilerStop();
#endif

    auto end = std::chrono::steady_clock::now();
    size_t mem_after = get_mem_usage();

    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;

    size_t mem = sizeof(kdtree->m_data); // outer vector object
    for (const auto& vec : kdtree->m_data) {
        mem += sizeof(vec); // already includes all doubles
    }

    std::cout << "Index Size: " << (mem_after - mem_before) + mem << " bytes\n";

}

~KDTree() {
    delete kdtree;
}

Points knn_query(Point& q, unsigned int k) {
    const size_t num_of_results = k;
    std::vector<size_t> ret_indexes(k);
    std::vector<double> out_dist_sqr(k);

    auto start = std::chrono::steady_clock::now();
    kdtree->query_knn(&q[0], num_of_results, &ret_indexes[0], &out_dist_sqr[0]);
    Points result;
    result.reserve(num_of_results);
    for (auto idx : ret_indexes) {
        result.emplace_back(kdtree->m_data[idx]);
    }
    auto end = std::chrono::steady_clock::now();

    knn_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    knn_count ++;
    kdtree->index->root_node_;
    return result;
}


inline Points range_query(Box& box) {

    point_t<Dim> min = box.min_corner();
    point_t<Dim> max = box.max_corner();

    using index_t = typename kdtree_t::index_t;
    typename index_t::BoundingBox query_box;

    for (size_t d = 0; d < Dim; ++d)
    {
        query_box[d].low  = min[d];
        query_box[d].high = max[d];
    }
    
    // std::vector<size_t> ret_indexes(kdtree->kdtree_get_point_count());
    // std::vector<double>    out_dists_sqr(kdtree->kdtree_get_point_count());

    // auto start = std::chrono::steady_clock::now();
    // // call low-level query
    // size_t num_of_results = kdtree->query_range(query_box, kdtree->kdtree_get_point_count(), &ret_indexes[0], &out_dists_sqr[0]);
    // auto end = std::chrono::steady_clock::now();

    
    // ###### Version 2 ######
    std::vector<size_t> ret_indexes;

    auto start = std::chrono::steady_clock::now();
    // call low-level query
    size_t num_of_results = kdtree->query_range_v2(query_box, ret_indexes);
    auto end = std::chrono::steady_clock::now();


    Points result;
    result.reserve(num_of_results);
    for (size_t i = 0; i < num_of_results; ++i){
        size_t idx = ret_indexes[i];
        result.emplace_back(kdtree->m_data[idx]);
    }

    range_count++;
    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return result;
}


inline size_t count() {
    return kdtree->kdtree_get_point_count();
}

inline size_t get_mem_usage() {
    struct mallinfo2 mi = mallinfo2();
    return mi.uordblks; // total bytes currently allocated
}

private:
// internal kdtree using nanoflann
kdtree_t* kdtree;

};

}}

