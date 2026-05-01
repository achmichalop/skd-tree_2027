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
class AG : public BaseIndex {

using Point = point_t<dim>;
using Points = std::vector<Point>;
using Range = std::pair<size_t, size_t>;
using Box = box_t<dim>;

public:
    AG(Points& points) {
        auto start = std::chrono::steady_clock::now();

        this->num_of_points = points.size();
        size_t num_of_partitions = ceil(this->num_of_points/128);
        size_t expected_num_of_partitions_per_d = round(std::pow(num_of_partitions, 1.0 / dim));

        Points sample;
        get_sample(points, sample);

        num_of_partitions = 1;
        for (size_t i = 0; i < dim; i++)
        {
            extractPartitionsPerDim(sample, i, expected_num_of_partitions_per_d);
            num_of_partitions *= this->numPartitionsPerDim[i];
            this->dim_offset[i] = 1;

            for (size_t j = 0; j < i; j++)
            {
                this->dim_offset[i] *= this->numPartitionsPerDim[j];
            }
        }
        
        buckets.resize(num_of_partitions);

        
        // insert points to buckets
        for (auto p : points) {
            //std::cout<<"compute_id(p) = " << compute_id(p) <<" "<< buckets.size() <<std::endl;
            buckets[compute_id(p)].emplace_back(p);
        }

        bounding_boxes.resize(num_of_partitions);

        for (size_t i = 0; i < num_of_partitions; ++i) {

            Point min_corner;
            Point max_corner;

            for (size_t j = 0; j < dim; ++j) {

                // recover per-dimension index from linear id
                size_t idx_j = (i / dim_offset[j]) % numPartitionsPerDim[j];

                // compute bounds
                min_corner[j] = (idx_j > 0) ? partitionsPerDim[j][idx_j-1] : 0.0f;
                max_corner[j] = partitionsPerDim[j][idx_j];
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
        ranges.emplace_back(std::make_pair(get_dim_idx(0, box.min_corner()[0]), get_dim_idx(0, box.max_corner()[0])));

        // find all intersect ranges
        for (size_t i=1; i<dim; ++i) {
            auto start_idx = get_dim_idx(i, box.min_corner()[i]);
            auto end_idx = get_dim_idx(i, box.max_corner()[i]);

            std::vector<Range> temp_ranges;
            for (auto idx=start_idx; idx<=end_idx; ++idx) {
                for (size_t j=0; j<ranges.size(); ++j) {
                    temp_ranges.emplace_back(std::make_pair(ranges[j].first + idx*dim_offset[i], ranges[j].second + idx*dim_offset[i]));
                }
            }

            // update the range vector
            ranges = temp_ranges;
        }

        // Points candidates;
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
    
    inline size_t get_mem_usage() {
        struct mallinfo2 mi = mallinfo2();
        return mi.uordblks; // total bytes currently allocated
    }


private:
    double num_of_points;

    std::vector<Points> buckets;
    std::vector<Box> bounding_boxes;
    std::array<size_t, dim> dim_offset;
    std::array<size_t, dim> numPartitionsPerDim;
    std::array<std::vector<double>, dim> partitionsPerDim;


    inline size_t get_dim_idx(size_t currDim, double coordinate)
    {

        if (coordinate <= 0.0f)
            return 0;
        else if (coordinate >= 1.0f)
            return this->numPartitionsPerDim[currDim]-1;
        size_t low = 0;
        size_t high = this->numPartitionsPerDim[currDim];
        size_t mid;
        
        while(low<high)
        {
            mid = (low+high)/2;
            if (coordinate > this->partitionsPerDim[currDim][mid]) low = mid+1;
            else high = mid;
        }
        return high;
    }

    // compute the bucket ID of a given point
    inline size_t compute_id(Point& p) {
        size_t id = 0;

        for (size_t i=0; i<dim; ++i) {
            auto current_idx = get_dim_idx(i, p[i]);
            id += current_idx * dim_offset[i];
        }

        return id;
    }

    void get_sample(Points points, Points& sample)
    {
        size_t total = points.size();
        size_t sample_size = total * 0.1;

        std::vector<size_t> sampleIdx;
        sampleIdx.reserve(sample_size);
        size_t step = std::max<size_t>(1, total / sample_size);
        for (size_t i = 0; i < total && sampleIdx.size() < sample_size; i += step)
            sampleIdx.push_back(i);
        
        
        for (size_t i = 0; i < sample_size; i++)
        {
            sample.emplace_back(points[sampleIdx[i]]);
        }
    }

    void extractPartitionsPerDim(Points& points, size_t currDim, size_t expectedPartitions)
    {
        size_t lowerBoundPos = 0;
        size_t upperBoundPos = points.size();
        size_t depth = ceil(log2((double) expectedPartitions));

        uint32_t step = ceil(upperBoundPos / (double) expectedPartitions);

        std::vector<size_t> bounds;
        std::vector<double> temp_partitions;
        bounds.resize(1 << depth);
        bounds[0] = lowerBoundPos;
        fill(bounds.begin() + expectedPartitions - 1, bounds.end(), upperBoundPos);

        temp_partitions.resize(1 << depth, 1.0f);

        for (size_t d = depth; d > 0; d--)
        {
            size_t low = 0;
            size_t stride = 1 << d;
            size_t high = stride - 1;
            size_t numIterations = 1 << (depth - d);
            
            for (size_t j = 0; j < numIterations; j++)
            {
                lowerBoundPos = bounds[low];
                upperBoundPos = bounds[high];
                size_t examinedPartPos = low + ((high - low) >> 1);

                if (examinedPartPos >= (expectedPartitions - 1))
                    break;

                size_t medianElemPos = (examinedPartPos + 1) * step;
                
                if (!temp_partitions[high])
                {
                    temp_partitions[examinedPartPos] = 0.0f;
                    bounds[examinedPartPos] = medianElemPos;
                    low = high;
                    high += stride;
                    continue;
                }

                auto lowIt = points.begin() + lowerBoundPos;
                auto medianIt = points.begin() + medianElemPos;
                auto highIt = points.begin() + upperBoundPos;

                nth_element(lowIt, medianIt, highIt, [currDim](auto& a, auto& b){return a[currDim] < b[currDim];});
        
                double examinedSeparator = points[medianElemPos][currDim];

                auto partitionIt = partition(lowIt, highIt, [examinedSeparator, currDim](auto& p){return p[currDim] < examinedSeparator;});

                uint32_t counter = partitionIt - points.begin();

                if (counter == lowerBoundPos)
                {
                    examinedSeparator = 0;
                    counter = medianElemPos;
                }

                temp_partitions[examinedPartPos] = examinedSeparator;
                bounds[examinedPartPos] = counter;

                low = high;
                high += stride;
            }
        }

        fill(temp_partitions.begin() + expectedPartitions, temp_partitions.end(), 0.0f);

        this->numPartitionsPerDim[currDim] = temp_partitions.size() - std::count(temp_partitions.begin(), temp_partitions.end(), 0.0f);
        this->partitionsPerDim[currDim].resize(this->numPartitionsPerDim[currDim]);
        for (size_t i = 0, k = 0; i < temp_partitions.size(); i++)
        {
            if (temp_partitions[i] != 0.0f)
                this->partitionsPerDim[currDim][k++] = temp_partitions[i];
        }

    }

    inline size_t index_size() const {
        size_t bytes = 0;

        /* ============================
        Buckets: vector<Points>
        ============================ */
        bytes += buckets.size() * sizeof(Points); // outer vector

        for (const auto& bucket : buckets) {
            bytes += bucket.size() * sizeof(Point); // stored points
        }

        /* ============================
        Bounding boxes
        ============================ */
        // bytes += bounding_boxes.capacity() * sizeof(Box);

        /* ============================
        Partition metadata
        ============================ */
        bytes += sizeof(dim_offset);
        bytes += sizeof(numPartitionsPerDim);

        /* ============================
        partitionsPerDim
        array<vector<double>, dim>
        ============================ */
        bytes += sizeof(partitionsPerDim); // array itself

        for (size_t i = 0; i < dim; ++i) {
            bytes += partitionsPerDim[i].capacity() * sizeof(double);
        }

        /* ============================
        Scalar fields
        ============================ */
        bytes += sizeof(num_of_points);

        /* ============================
        (Optional) object header
        ============================ */
        // bytes += sizeof(*this);

        return bytes;
    }

};

}
}
