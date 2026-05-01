#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <tpie/tpie.h>
#include <utility>
#include <vector>
#include "common.hpp"
#include "type.hpp"

namespace bench { namespace utils { 
    enum OperationType {INSERT = 0, DELETE, RANGE , KNN}; 
    template <size_t dim>
    void read_points(vec_of_point_t<dim>& out_points, const std::string& fname, const size_t N)  
    {
        out_points.clear();
        out_points.reserve(N);

        out_points.resize(N);


        std::ifstream in(fname, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot open binary file");

        in.read(reinterpret_cast<char*>(out_points.data()), N * dim * sizeof(double));

        std::array<double, dim> minCoordinates;
        std::array<double, dim> maxCoordinates;
        minCoordinates.fill(std::numeric_limits<double>::max());
        maxCoordinates.fill(-std::numeric_limits<double>::max());

    #pragma omp parallel
        {
            std::array<double, dim> localMin = minCoordinates;
            std::array<double, dim> localMax = maxCoordinates;

    #pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i)
            {
    #pragma omp simd
                for (size_t j = 0; j < dim; ++j)
                {
                    double v = out_points[i][j];
                    localMin[j] = std::min(localMin[j], v);
                    localMax[j] = std::max(localMax[j], v);
                }
            }

    #pragma omp critical
            {
    #pragma omp simd
                for (size_t j = 0; j < dim; ++j)
                {
                    minCoordinates[j] = std::min(minCoordinates[j], localMin[j]);
                    maxCoordinates[j] = std::max(maxCoordinates[j], localMax[j]);
                }
            }
        }

    #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
        {
    #pragma omp simd
            for (size_t j = 0; j < dim; ++j)
            {
                double extent = maxCoordinates[j] - minCoordinates[j];
                out_points[i][j] = (extent > 0.0) ? (out_points[i][j] - minCoordinates[j]) / extent : 0.0;
            }
        }
    }

    template <size_t dim>
    void read_box_queries(std::vector<std::pair<size_t, box_t<dim>>> &box_queries, const std::string& fname, const size_t N_queries)
    {
        box_queries.reserve(N_queries);

        std::ifstream in(fname);
        if (!in.is_open())
            throw std::runtime_error("Could not open file: " + fname);

        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream strm(line);

            point_t<dim> min_p{};
            point_t<dim> max_p{};

            // Read min coordinates
            for (size_t i = 0; i < dim; ++i)
            {
                char comma;
                strm >> min_p[i];
                if (i + 1 < dim) strm >> comma; // consume ','
            }

            // Skip whitespace between min and max parts
            strm >> std::ws;

            // Read max coordinates
            for (size_t i = 0; i < dim; ++i)
            {
                char comma;
                strm >> max_p[i];
                if (i + 1 < dim) strm >> comma; // consume ','
            }

            box_t<dim> box(min_p, max_p);
            box_queries.emplace_back(make_pair(RANGE, box));

            if (box_queries.size() == N_queries)
                break;
        }

    }

    template <size_t dim>
    void read_knn_queries(std::vector<std::pair<size_t, box_t<dim>>>& knn_queries, const std::string& fname, const size_t N_queries)
    {
        knn_queries.reserve(N_queries);

        std::ifstream in(fname);
        if (!in.is_open())
            throw std::runtime_error("Could not open file: " + fname);

        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream strm(line);

            point_t<dim> p{};
            // Read min coordinates
            for (size_t i = 0; i < dim; ++i)
            {
                char comma;
                strm >> p[i];
                if (i + 1 < dim) strm >> comma; // consume ','
            }

            box_t<dim> box(p, p);
            knn_queries.emplace_back(make_pair(KNN, box));

            if (knn_queries.size() == N_queries)
                break;
        }
    }

    template <size_t dim>
    void read_points_construction_updates(vec_of_point_t<dim>& construction_points, const std::string& fname, const size_t N, std::vector<std::pair<size_t, box_t<dim>>>& insertions, std::vector<std::pair<size_t, box_t<dim>>>& deletions, double insertions_ratio)  
    {

        vec_of_point_t<dim> out_points;
        out_points.reserve(N);

        out_points.resize(N);

        std::ifstream in(fname, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot open binary file");

        in.read(reinterpret_cast<char*>(out_points.data()), N * dim * sizeof(double));

        std::array<double, dim> minCoordinates;
        std::array<double, dim> maxCoordinates;
        minCoordinates.fill(std::numeric_limits<double>::max());
        maxCoordinates.fill(-std::numeric_limits<double>::max());

        // --- Compute min/max in parallel ---
    #pragma omp parallel
        {
            std::array<double, dim> localMin = minCoordinates;
            std::array<double, dim> localMax = maxCoordinates;

    #pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i)
            {
    #pragma omp simd
                for (size_t j = 0; j < dim; ++j)
                {
                    double v = out_points[i][j];
                    localMin[j] = std::min(localMin[j], v);
                    localMax[j] = std::max(localMax[j], v);
                }
            }

    #pragma omp critical
            {
    #pragma omp simd
                for (size_t j = 0; j < dim; ++j)
                {
                    minCoordinates[j] = std::min(minCoordinates[j], localMin[j]);
                    maxCoordinates[j] = std::max(maxCoordinates[j], localMax[j]);
                }
            }
        }

        // --- Normalize points in parallel ---
    #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
        {
    #pragma omp simd
            for (size_t j = 0; j < dim; ++j)
            {
                double extent = maxCoordinates[j] - minCoordinates[j];
                out_points[i][j] = (extent > 0.0)
                                    ? (out_points[i][j] - minCoordinates[j]) / extent
                                    : 0.0;  // avoid division by zero
            }
        }


        size_t numOfInsertions = static_cast<size_t>(N * insertions_ratio);
        numOfInsertions -= (numOfInsertions % 5);
        size_t numOfDeletions = static_cast<size_t>(numOfInsertions / 5);
        size_t numOfConstruction = N - numOfInsertions;
        size_t stepInsertions = numOfConstruction / numOfInsertions;

        if (stepInsertions == 0)
        {
            std::cout << "We cannot cut a workload!!!" << std::endl;
            exit(-1);
        }

        construction_points.reserve(numOfConstruction);
        insertions.reserve(numOfInsertions);
        deletions.reserve(numOfDeletions);
        
        size_t counter = 0;
        for (size_t idx = 0; idx < out_points.size(); idx++) {
            if ((idx % stepInsertions) == 0 && insertions.size() < numOfInsertions) {
                box_t<dim> box(out_points[idx], out_points[idx]);
                insertions.emplace_back(make_pair(INSERT, box));
            } else {
                construction_points.emplace_back(out_points[idx]);
            }
        }

        // constant shuffle of updates_points
        std::mt19937 rng(12345); // fixed seed for reproducibility
        std::shuffle(insertions.begin(), insertions.end(), rng);

        size_t stepDeletions = construction_points.size() / numOfDeletions;

        for (size_t i = 0; i < numOfDeletions; ++i)
        {
            const auto& p = construction_points[i * stepDeletions];
            box_t<dim> box(p, p);
            deletions.emplace_back(DELETE, box);
        }

        std::shuffle(deletions.begin(), deletions.end(), rng);
    }

    template <size_t dim>
    void merge_updates(std::vector<std::pair<size_t, box_t<dim>>>& insertions, std::vector<std::pair<size_t, box_t<dim>>>& deletions, std::vector<std::pair<size_t, box_t<dim>>>& merged_updates)
    {
        if (insertions.empty())
            return;

        if (deletions.size() * 5 != insertions.size())
            throw std::runtime_error("Invalid insertion/deletion ratio");

        merged_updates.reserve(insertions.size() + deletions.size());
        std::cout << "Insertions: " << insertions.size() << " -- Deletions: " << deletions.size() << std::endl;
        size_t del_idx = 0;
        constexpr size_t deletions_per_insertion = 1; // 1 per 5 insertions
        constexpr size_t insertion_block = 5;

        for (size_t ins_idx = 0; ins_idx < insertions.size(); ++ins_idx)
        {
            // Append insertion
            merged_updates.push_back(insertions[ins_idx]);

            // After every 5 insertions, append 1 deletion
            if ((ins_idx + 1) % insertion_block == 0 &&
                del_idx < deletions.size())
            {
                merged_updates.push_back(deletions[del_idx++]);
            }
        }
    }

    template <size_t dim>
    void merge_queries(std::vector<std::pair<size_t, box_t<dim>>>& box_queries, std::vector<std::pair<size_t, box_t<dim>>>& knn_queries, std::vector<std::pair<size_t, box_t<dim>>>& merged_queries)
    {
        merged_queries.reserve(box_queries.size() + knn_queries.size());

        for (size_t idx = 0; idx < box_queries.size(); ++idx)
        {
            // Append box query
            merged_queries.push_back(box_queries[idx]);
            // Append box query
            merged_queries.push_back(knn_queries[idx]);
        }
    }

    template <size_t dim>
    void make_workload(std::vector<std::pair<size_t, box_t<dim>>>& merged_updates, std::vector<std::pair<size_t, box_t<dim>>>& merged_queries, std::vector<std::pair<size_t, box_t<dim>>>& final_workload)
    {
        size_t parts = 5;

        final_workload.clear();
        final_workload.reserve(
            merged_updates.size() + parts * merged_queries.size());

        const size_t base_chunk = merged_updates.size() / parts;
        size_t offset = 0;

        for (size_t i = 0; i < parts; ++i)
        {
            // Last chunk takes the remainder
            size_t current_chunk_size =
                (i == parts - 1) ? (merged_updates.size() - offset) : base_chunk;

            // Append chunk of merged_updates
            final_workload.insert(final_workload.end(),
                                merged_updates.begin() + offset,
                                merged_updates.begin() + offset + current_chunk_size);

            // Append full merged_queries
            final_workload.insert(final_workload.end(),
                                merged_queries.begin(),
                                merged_queries.end());

            offset += current_chunk_size;
        }
    }

}
}