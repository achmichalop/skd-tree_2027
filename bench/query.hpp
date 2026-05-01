#pragma once

#include <cstddef>
#include <random>
#include <map>
#include <algorithm>

#include "../utils/type.hpp"
#include "../utils/datautils.hpp"
#include "../indices/nonlearned/fullscan.hpp"


namespace bench { namespace query {

template<class Index, size_t Dim>
static void custom_batch_range_queries(Index& index, std::vector<std::pair<size_t, box_t<Dim>>> &range_queries) {
    
    std::vector<std::pair<long, size_t>> range_time;
    range_time.reserve(range_queries.size());
    
    for (auto& query : range_queries) {
        box_t<Dim> box = query.second;
        auto results = index.range_query(box);
        range_time.emplace_back(index.get_range_time(), results.size());
        index.reset_timer();
    }

    // sort by range_cnt
    std::sort(range_time.begin(), range_time.end());

    long long total_time = 0;
    long long total_results = 0;

    for (auto& entry : range_time)
    {
        total_time += entry.first;
        total_results += entry.second;
    }

    std::cout << "Fast Query Time : " << range_time[0].first << " [us] -- results: " << range_time[0].second << std::endl;
    std::cout << "Slow Query Time : " << range_time[range_time.size()-1].first << " [us] -- results: " << range_time[range_time.size()-1].second << std::endl;
    std::cout << "Avg Query Time: " << total_time / (double) range_time.size() << " [us] -- results Avg results: " << total_results / (double) range_time.size() << std::endl;

    double avg_selectivity = (total_results / (double) range_time.size()) / index.count();
    long long throughput = round((range_time.size() * 1000000) / (double) total_time);
    std::cout << "Selectivity " << avg_selectivity * 100 << "% -- throughput: " << throughput << " queries/sec" << std::endl;
}

template<class Index, size_t Dim>
static void custom_batch_knn_queries_var_k(Index& index, std::vector<std::pair<size_t, box_t<Dim>>>& knn_queries) {
    int ks[6] = {1, 5, 10, 50, 100, 500};

    for (auto k : ks) {
        std::cout << "For k = " << k <<std::endl;
        std::vector<long> knn_time;

        for (auto& query : knn_queries) {
            point_t<Dim> q_point = query.second.min_corner();

            index.knn_query(q_point, k);
            knn_time.emplace_back(index.get_knn_time());
            index.reset_timer();
        }
        std::sort(knn_time.begin(), knn_time.end());
        long long total_time = 0;
        for (auto& entry : knn_time)
        {
            total_time += entry;
        }

        std::cout << "\tFast Query Time : " << knn_time[0] / (double)1000 << " [us]" << std::endl;
        std::cout << "\tSlow Query Time : " << knn_time[knn_time.size()-1] / (double)1000 << " [us]" << std::endl;
        std::cout << "\tAvg Query Time: " << total_time / (double) (knn_time.size()*1000) << " [us]" << std::endl;
        long long throughput = round((knn_time.size() * 1000000000) / (double)total_time);
        std::cout <<"\tThroughput: " << throughput << " queries/sec" << std::endl;
        std::cout << "###################################\n\n";
        knn_time.clear();
    }

}

template<class Index, size_t Dim>
static void custom_batch_mixed_queries(Index& index, std::vector<std::pair<size_t, box_t<Dim>>> mixed_queries) {

    unsigned long knn_time = 0, range_time = 0, insert_time = 0, remove_time = 0;

    for (auto& q : mixed_queries)
    {
        size_t type = q.first;
        switch (type)
        {
            case bench::utils::INSERT:
            {  
                point_t<Dim> point = q.second.min_corner();
                index.insert(point);
                insert_time += index.get_updates_time();
                index.reset_timer();
                break;
            }
            case bench::utils::DELETE:
            {
                point_t<Dim> point = q.second.min_corner();
                index.remove(point);
                remove_time += index.get_removes_time();
                index.reset_timer();
                break;
            }
            case bench::utils::RANGE:
            {
                box_t<Dim> box = q.second;
                index.range_query(box);
                range_time += index.get_range_time();
                index.reset_timer();
                break;
            }
            case bench::utils::KNN:
            {
                point_t<Dim> point = q.second.min_corner();
                index.knn_query(point, 10);
                knn_time += index.get_knn_time();
                index.reset_timer();
                break;
            }
        }
    }

    std::cout << "\nInserts Time: " << insert_time / (double) 1000000 << " [s]" << std::endl;
    std::cout << "Removes Time: " << remove_time / (double) 1000000 << " [s]" << std::endl;
    std::cout << "Queries Times: " << (range_time / (double) 1000000) + (knn_time / (double) 1000000000) << " [s]" << std::endl;
    double total_time =  (insert_time / (double) 1000000) + (remove_time / (double) 1000000) + (range_time / (double) 1000000) + (knn_time / (double) 1000000000);
    std::cout << "Total Time: " << total_time << " [s]" << std::endl;


}
}
}

