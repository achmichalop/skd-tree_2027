#pragma once

#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../base_index.hpp"

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


namespace bench { namespace index {

// uniform K by K ... grid 
template<size_t dim>
class UG : public BaseIndex {

using Point = point_t<dim>;
using Points = std::vector<Point>;
using Range = std::pair<size_t, size_t>;
using Box = box_t<dim>;

public:
    UG(Points& points) {
        auto start = std::chrono::steady_clock::now();

        this->num_of_points = points.size();

        int num_of_partitions = ceil(this->num_of_points/128);
        num_of_partitions_per_d = round(std::pow(num_of_partitions, 1.0 / dim));        
        num_of_partitions = std::pow(num_of_partitions_per_d,dim);
        buckets.resize(num_of_partitions);

        std::cout << "part per dim: " << num_of_partitions_per_d << std::endl;
        std::cout << "total part: " << num_of_partitions << std::endl;

        // dimension offsets when computing bucket ID
        for (size_t i=0; i<dim; ++i) {
            this->dim_offset[i] = bench::common::ipow(num_of_partitions_per_d, i);
        }

        // boundaries of each dimension
        std::fill(mins.begin(), mins.end(), std::numeric_limits<double>::max());
        std::fill(maxs.begin(), maxs.end(), std::numeric_limits<double>::min());


        for (size_t i=0; i<dim; ++i) {
            for (auto& p : points) {
                mins[i] = std::min(p[i], mins[i]);
                maxs[i] = std::max(p[i], maxs[i]);
            }
        }

        // widths of each dimension
        for (size_t i=0; i<dim; ++i) {
            widths[i] = (maxs[i] - mins[i]) / num_of_partitions_per_d;
        }

        
        // insert points to buckets
        for (auto p : points) {
            buckets[compute_id(p)].emplace_back(p);
        }


        bounding_boxes.resize(num_of_partitions);

        for (size_t i = 0; i < num_of_partitions; ++i) {

            Point min_corner;
            Point max_corner;

            for (size_t j = 0; j < dim; ++j) {

                // recover per-dimension index from linear id
                size_t idx_j = (i / dim_offset[j]) % num_of_partitions_per_d;

                // compute bounds
                min_corner[j] = mins[j] + idx_j * widths[j];
                max_corner[j] = min_corner[j] + widths[j];

            }

            bounding_boxes[i] = Box(min_corner, max_corner);
        }

        auto end = std::chrono::steady_clock::now();        
        build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
        std::cout << "Index Size: " << index_size() /(double) (1024*1024*1024) << std::endl;
    }


    Points range_query(Box& box) {
        auto start = std::chrono::steady_clock::now();

        // bucket ranges that intersect the query box
        std::vector<Range> ranges;

        // search range on the 1-st dimension
        ranges.emplace_back(std::make_pair(get_dim_idx(box.min_corner(), 0), get_dim_idx(box.max_corner(), 0)));

        // find all intersect ranges
        for (size_t i=1; i<dim; ++i) {
            auto start_idx = get_dim_idx(box.min_corner(), i);
            auto end_idx = get_dim_idx(box.max_corner(), i);

            std::vector<Range> temp_ranges;
            for (auto idx=start_idx; idx<=end_idx; ++idx) {
                for (size_t j=0; j<ranges.size(); ++j) {
                    temp_ranges.emplace_back(std::make_pair(ranges[j].first + idx*dim_offset[i], ranges[j].second + idx*dim_offset[i]));
                }
            }

            // update the range vector
            ranges = temp_ranges;
        }

        Points result;

        // find candidate points
        int counter = 0;
        for (auto range : ranges) {
            counter++;

            auto start_idx = range.first;
            auto end_idx = range.second;

            for (auto idx=start_idx; idx<=end_idx; ++idx) {
                
                if (boost::geometry::covered_by(bounding_boxes[idx], box)){
                    for (auto cand : this->buckets[idx]) 
                    {
                        result.emplace_back(cand);
                    }
                }
                else{
                    for (auto cand : this->buckets[idx]) 
                    {
                    
                    if (boost::geometry::covered_by(cand, box)){
                            result.emplace_back(cand);
                        }
                    }
                }
                
            }
        }

        auto end = std::chrono::steady_clock::now();
        range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        range_count ++;
        
        return result;
    }

    inline size_t count() {
        return this->num_of_points;
    }

    inline size_t index_size() const {
        size_t bytes = 0;

        /* ---- buckets: vector<Points> ---- */
        bytes += buckets.size() * sizeof(Points); // outer vector

        for (const auto& bucket : buckets) {
            bytes += bucket.size() * sizeof(Point); // actual stored points
        }

        /* ---- bounding boxes ---- */
        // bytes += bounding_boxes.capacity() * sizeof(Box);

        /* ---- inline arrays ---- */
        bytes += sizeof(mins);
        bytes += sizeof(maxs);
        bytes += sizeof(widths);
        bytes += sizeof(dim_offset);

        /* ---- object itself (optional, usually excluded) ---- */
        // bytes += sizeof(*this);

        return bytes;
    }



private:
    double num_of_points;
    size_t num_of_partitions_per_d;
    //std::array<Points, common::ipow(K, dim)> buckets;
    std::vector<Points> buckets;
    std::vector<Box> bounding_boxes;
    
    std::array<double, dim> mins;
    std::array<double, dim> maxs;
    std::array<double, dim> widths;
    std::array<size_t, dim> dim_offset;

    // compute the index on d-th dimension of a given point
    inline size_t get_dim_idx(Point& p, const size_t& d) {
        if (p[d] <= mins[d]) {
            return 0;
        } else if (p[d] >= maxs[d]) {
            return num_of_partitions_per_d-1;
        } else {
            return (size_t) ((p[d] - mins[d]) / widths[d]);
        }
    }

    // compute the bucket ID of a given point
    inline size_t compute_id(Point& p) {
        size_t id = 0;

        for (size_t i=0; i<dim; ++i) {
            auto current_idx = get_dim_idx(p, i);
            id += current_idx * dim_offset[i];
        }

        return id;
    }

};

}
}
