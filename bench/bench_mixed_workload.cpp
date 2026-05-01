#include "../utils/datautils.hpp"
#include "../utils/common.hpp"

#include "../indices/nonlearned/nonlearned_index.hpp"
#include "../indices/learned/learned_index.hpp"

#include "query.hpp"

#include <cstddef>
#include <string>
#include <stdlib.h>

#ifndef BENCH_DIM
#define BENCH_DIM 2
#endif


using Point = point_t<BENCH_DIM>;
using Points = std::vector<point_t<BENCH_DIM>>;
using Box = box_t<BENCH_DIM>;

// non-learned indices
using SKDTree = bench::index::SKDTREE<BENCH_DIM>;
using RTree = bench::index::RTree<BENCH_DIM>;
using PHTree = bench::index::PH_TREE<BENCH_DIM>;

// linear scan
using FS = bench::index::FullScan<BENCH_DIM>;

struct IndexSet {
    SKDTree* skdtree;
    RTree*     rtree;
    PHTree* phtree;
    FS*        fs;

    IndexSet() : 
        skdtree(nullptr),
        rtree(nullptr), 
        phtree(nullptr),
        fs(nullptr){}

    ~IndexSet() {
        delete skdtree;
        delete rtree;
        delete phtree;
        delete fs;
    }
};


static void build_index(IndexSet& idx_set, const std::string& idx_name, Points& points) {
    if (idx_name.compare("skdtree") == 0) {//
        idx_set.skdtree = new SKDTree(points);
        return;
    }

    if (idx_name.compare("rtree") == 0) {///////
        idx_set.rtree = new RTree(points);
        return;
    }

    if (idx_name.compare("phtree") == 0) {
        idx_set.phtree = new PHTree(points);
        return;
    }

    if (idx_name.compare("fs") == 0) {///
        idx_set.fs = new FS(points);
        return;
    }

    exit(0);
}


int main(int argc, char **argv) {

    std::string index = argv[1]; // index name
    std::string fname = argv[2]; // data file name
    size_t N = std::stoi(argv[3]); // dataset size
    double insertions_ratio = std::stof(argv[4]); // insertions ratio / deletions will be 1/5 of insertions
    std::string box_fname = argv[5];
    std::string knn_fname = argv[6];
    size_t queries_N = std::stoi(argv[7]); // queries size

    std::cout << "====================================" << std::endl;
    std::cout << "Load data: " << fname << std::endl;

    Points construction_points;
    std::vector<std::pair<size_t, Box>> insertions, deletions, merged_updates;
    std::vector<std::pair<size_t, Box>> box_queries, knn_queries, merged_queries, final_workload;
    
    bench::utils::read_points_construction_updates(construction_points, fname, N, insertions, deletions, insertions_ratio);
    bench::utils::read_box_queries(box_queries, box_fname, queries_N);
    bench::utils::read_knn_queries(knn_queries, knn_fname, queries_N);
    bench::utils::merge_updates(insertions, deletions, merged_updates);
    bench::utils::merge_queries(box_queries, knn_queries, merged_queries);
    bench::utils::make_workload(merged_updates, merged_queries, final_workload);

    IndexSet idx_set;
    
    build_index(idx_set, index, construction_points);

    if (index.compare("skdtree") == 0) {
        assert(idx_set.skdtree != nullptr);
        bench::query::custom_batch_mixed_queries(*(idx_set.skdtree), final_workload);
        return 0;
    }

    if (index.compare("rtree") == 0) {
        assert(idx_set.rtree != nullptr);
        bench::query::custom_batch_mixed_queries(*(idx_set.rtree), final_workload);
        return 0;
    }

    if (index.compare("phtree") == 0) {
        assert(idx_set.phtree != nullptr);
        bench::query::custom_batch_mixed_queries(*(idx_set.phtree), final_workload);
        return 0;
    }
}
