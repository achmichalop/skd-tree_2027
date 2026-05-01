#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <variant>
#include <boost/multi_array.hpp>
#include <vector>
#include <chrono>
#include <random>
#include "../base_index.hpp"
#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../pgm/pgm_index.hpp"
#include "../pgm/pgm_index_variants.hpp"


namespace bench { namespace index {

// the sort dimension is always the last dimension
template<size_t Dim, size_t Eps=64, size_t SortDim=Dim-1>
class Flood : public BaseIndex {

using Point = point_t<Dim>;
using Points = std::vector<Point>;
using Range = std::pair<size_t, size_t>;
using Box = box_t<Dim>;

using Index = pgm::PGMIndex<double, Eps>;

public:

class Bucket {
    public:
    Points _local_points;
    // eps for each bucket is fixed to 16 based on a micro benchmark
    pgm::PGMIndex<double, 16>* _local_pgm;

    Bucket() : _local_pgm(nullptr) {} ;

    ~Bucket() {
        delete this->_local_pgm;
    }

    inline void insert(Point& p) {
        this->_local_points.emplace_back(p);
    }

    inline void build() {
        if (_local_points.size() == 0) {
            return;
        }
        // note points are already sorted by SortDim
        std::vector<double> idx_data;
        idx_data.reserve(_local_points.size());
        for (const auto& p : _local_points) {
            idx_data.emplace_back(std::get<SortDim>(p));
        }
        
        _local_pgm = new pgm::PGMIndex<double, 16>(idx_data);
    }

    inline void search(Points& result, Box& box) {
        if (_local_pgm == nullptr) {
            return;
        }
        
        auto min_key = std::get<SortDim>(box.min_corner());
        auto max_key = std::get<SortDim>(box.max_corner());
        auto range_lo = this->_local_pgm->search(min_key);
        auto range_hi = this->_local_pgm->search(max_key);

        for (size_t i=range_lo.lo; i<range_hi.hi; ++i) {
            if (bench::common::is_in_box(this->_local_points[i], box)) {
                result.emplace_back(this->_local_points[i]);
            }
        }
    }
};

Flood(Points& points) : _data(points) {
    
    auto start = std::chrono::steady_clock::now();
    this->num_of_points = points.size();
    int num_of_partitions = ceil(this->num_of_points/2000);
    num_of_partitions_per_d = round(std::pow(num_of_partitions, 1.0 / (Dim-1)));
    bucket_size = ((num_of_points + num_of_partitions_per_d - 1)/num_of_partitions_per_d);

    // dimension offsets when computing bucket ID
    for (size_t i=0; i<Dim-1; ++i) {
        this->dim_offset[i] = bench::common::ipow(num_of_partitions_per_d, i);        
    }

    buckets.resize(bench::common::ipow(num_of_partitions_per_d, Dim-1));

    // sort points by SortDim
    std::sort(_data.begin(), _data.end(), [](auto& p1, auto& p2) {
        return std::get<SortDim>(p1) < std::get<SortDim>(p2);
    });

    // boundaries of each dimension
    std::fill(mins.begin(), mins.end(), std::numeric_limits<double>::max());
    std::fill(maxs.begin(), maxs.end(), std::numeric_limits<double>::min());

    // train model on dimension 1 -- Dim-1
    std::vector<double> idx_data;
    idx_data.reserve(points.size());
    for (size_t i=0; i<Dim-1; ++i) {
        for (const auto& p : _data) {
            mins[i] = std::min(p[i], mins[i]);
            maxs[i] = std::max(p[i], maxs[i]);

            idx_data.emplace_back(p[i]);
        }

        std::sort(idx_data.begin(), idx_data.end());
        this->indexes[i] = new Index(idx_data);

        idx_data.clear();
    }


    // note data are sorted by SortDim
    for (auto& p : _data) {
        buckets[compute_id(p)].insert(p);
    }

    for (auto& b : buckets) {
        b.build();
    }

    auto end = std::chrono::steady_clock::now();

    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
    std::cout << "Index Size: " << index_size() /(double) (1024*1024*1024) << std::endl;

}

Points range_query(Box& box) {

    auto start = std::chrono::steady_clock::now();

    // find all intersected cells
    std::vector<std::pair<size_t, size_t>> ranges;
    find_intersect_ranges(ranges, box);
    
    // search each cell using local models
    Points result;
    for (auto& range : ranges) {
        for (auto idx=range.first; idx<=range.second; ++idx) {
            this->buckets[idx].search(result, box);
        }
    }


    auto end = std::chrono::steady_clock::now();
    
    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    range_count ++;

    return result;
}

inline size_t count() {
    return _data.size();
}

inline size_t index_size() const {
    size_t bytes = 0;
    for (size_t i = 0; i < Dim - 1; ++i) {
        if (indexes[i]) {
            bytes += indexes[i]->size_in_bytes();
        }
    }

    bytes += buckets.size() * sizeof(Bucket);

    for (const auto& b : buckets) {
        bytes += b._local_points.size() * sizeof(Point);

        if (b._local_pgm) {
            bytes += b._local_pgm->size_in_bytes();
        }
    }

    bytes += sizeof(dim_offset);
    bytes += sizeof(mins);
    bytes += sizeof(maxs);

    bytes += sizeof(num_of_points);
    bytes += sizeof(num_of_partitions_per_d);
    bytes += sizeof(bucket_size);

    return bytes;
}



inline size_t get_mem_usage() {
    struct mallinfo2 mi = mallinfo2();
    return mi.uordblks; 
}

~Flood() {
    for (size_t i=0; i<Dim-1; ++i) {
        delete this->indexes[i];
    }
}


private:
double num_of_points;
size_t num_of_partitions_per_d;

Points& _data;
std::array<Index*, Dim-1> indexes;
std::vector<Bucket> buckets;
std::array<size_t, Dim-1> dim_offset;

std::array<double, Dim-1> mins;
std::array<double, Dim-1> maxs;

size_t bucket_size;

inline void find_intersect_ranges(std::vector<std::pair<size_t, size_t>>& ranges, Box& qbox) {
    if (Dim == 2) {
        ranges.emplace_back(get_dim_idx(qbox.min_corner(), 0), get_dim_idx(qbox.max_corner(), 0));
    } else {
        // search range on the 1-st dimension
        ranges.emplace_back(get_dim_idx(qbox.min_corner(), 0), get_dim_idx(qbox.max_corner(), 0));
        
        // find all intersect ranges
        for (size_t i=1; i<Dim-1; ++i) {
            auto start_idx = get_dim_idx(qbox.min_corner(), i);
            auto end_idx = get_dim_idx(qbox.max_corner(), i);

            std::vector<std::pair<size_t, size_t>> temp_ranges;
            for (auto idx=start_idx; idx<=end_idx; ++idx) {
                for (size_t j=0; j<ranges.size(); ++j) {
                    temp_ranges.emplace_back(ranges[j].first + idx*dim_offset[i], ranges[j].second + idx*dim_offset[i]);
                }
            }

            // update the range vector
            ranges = temp_ranges;
        }
    }
}

// locate the bucket on d-th dimension using binary search
inline size_t get_dim_idx(Point& p, size_t d) {
    if (p[d] <= this->mins[d]) {
        return 0;
    }
    if (p[d] >= this->maxs[d]) {
        return num_of_partitions_per_d-1;
    }
    auto approx_pos = this->indexes[d]->search(p[d]).pos / this->bucket_size;
    return std::min(approx_pos, num_of_partitions_per_d-1);
}

inline size_t compute_id(Point& p) {
    size_t id = 0;

    for (size_t i=0; i<Dim-1; ++i) {
        auto current_idx = get_dim_idx(p, i);
        id += current_idx * dim_offset[i];
    }

    return id;
}


};
}
}
