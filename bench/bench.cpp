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

#ifndef INDEX_ERROR_THRESHOLD 
#define INDEX_ERROR_THRESHOLD 64
#endif

using Point = point_t<BENCH_DIM>;
using Points = std::vector<point_t<BENCH_DIM>>;
using Box = box_t<BENCH_DIM>;

// tree-based indices
using RTree = bench::index::RTree<BENCH_DIM>;
using KDTree = bench::index::KDTree<BENCH_DIM>;
using PHTree = bench::index::PH_TREE<BENCH_DIM>;
using SKDTree = bench::index::SKDTREE<BENCH_DIM>;

// grid indices
using UG = bench::index::UG<BENCH_DIM>;
using AG = bench::index::AG<BENCH_DIM>;

// learned indices
using IFI = bench::index::IFIndex<BENCH_DIM>;
using Flood = bench::index::Flood<BENCH_DIM, INDEX_ERROR_THRESHOLD>;

// linear scan
using FS = bench::index::FullScan<BENCH_DIM>;



struct IndexSet {
    SKDTree* skdtree;
    KDTree*    kdtree;
    RTree*     rtree;
    PHTree* phtree;
    AG*        ag;
    UG*        ug;
    Flood*     flood;
    IFI*       ifi;
    FS*        fs;

    IndexSet() : 
        skdtree(nullptr),
        kdtree(nullptr),
        rtree(nullptr),
        phtree(nullptr),
        ag(nullptr),
        ug(nullptr),
        flood(nullptr),
        ifi(nullptr),
        fs(nullptr){}

    ~IndexSet() {
        delete skdtree;
        delete kdtree;
        delete rtree;
        delete phtree;
        delete ag;
        delete ug;
        delete flood;
        delete ifi;
        delete fs;
    }
};


static void build_index(IndexSet& idx_set, const std::string& idx_name, Points& points) {
    if (idx_name.compare("skdtree") == 0) {//
        idx_set.skdtree = new SKDTree(points);
        return;
    }

    if (idx_name.compare("kdtree") == 0) {////
        idx_set.kdtree = new KDTree(points);
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

    if (idx_name.compare("ag") == 0) {////
        idx_set.ag = new AG(points);
        return;
    }

    if (idx_name.compare("ug") == 0) {////
        idx_set.ug = new UG(points);
        return;
    }

    if (idx_name.compare("flood") == 0) {
        idx_set.flood = new Flood(points);
        return;
    }


    if (idx_name.compare("ifi") == 0) {//
        idx_set.ifi = new IFI(points);
        return;
    }

    if (idx_name.compare("fs") == 0) {///
        idx_set.fs = new FS(points);
        return;
    }
    exit(0);
}

int main(int argc, char **argv) {
    assert(argc >= 4);

    std::string index = argv[1]; // index name
    std::string fname = argv[2]; // data file name
    size_t N = std::stoi(argv[3]); // dataset size
    std::string queries_fname = argv[4];
    size_t queries_N = std::stoi(argv[5]); // dataset size
    std::string mode = argv[6]; // bench mode {"range", "knn"}

    std::cout << "====================================" << std::endl;
    std::cout << "Load data: " << fname << std::endl;

    Points points;
    std::vector<std::pair<size_t, Box>> box_queries;
    std::vector<std::pair<size_t, Box>> knn_queries;
    IndexSet idx_set;


    bench::utils::read_points(points, fname, N);

    if (mode.compare("range") == 0)
    {
        bench::utils::read_box_queries(box_queries, queries_fname, queries_N);
    }
    else if (mode.compare("knn") == 0)
    {
        bench::utils::read_knn_queries(knn_queries, queries_fname, queries_N);
    }
    

    build_index(idx_set, index, points);

    if (index.compare("skdtree") == 0) {
        assert(idx_set.skdtree != nullptr);
        
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.skdtree), box_queries);
            return 0;
        }
        
        if (mode.compare("knn") == 0) {
            bench::query::custom_batch_knn_queries_var_k(*(idx_set.skdtree), knn_queries);
            return 0;
        }
    }

    if (index.compare("kdtree") == 0) {
        assert(idx_set.kdtree != nullptr);

        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.kdtree), box_queries);
            return 0;
        }

        if (mode.compare("knn") == 0) {
                bench::query::custom_batch_knn_queries_var_k(*(idx_set.kdtree), knn_queries);
            return 0;
        }
    }

    if (index.compare("rtree") == 0) {
        assert(idx_set.rtree != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.rtree), box_queries);
            return 0;
        }
        if (mode.compare("knn") == 0) {
            bench::query::custom_batch_knn_queries_var_k(*(idx_set.rtree), knn_queries);
            return 0;
        }
    }

    if (index.compare("phtree") == 0) {
        assert(idx_set.phtree != nullptr);

        if (mode.compare("range") == 0)
        {
            bench::query::custom_batch_range_queries(*(idx_set.phtree), box_queries);
            return 0;
        }
        if (mode.compare("knn") == 0) {
            bench::query::custom_batch_knn_queries_var_k(*(idx_set.phtree), knn_queries);
            return 0;
        }
    }
    

    if (index.compare("ag") == 0) {
        assert(idx_set.ag != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.ag), box_queries);
            return 0;
        }
    }

    if (index.compare("ug") == 0) {
        assert(idx_set.ug != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.ug), box_queries);
            return 0;
        }
    }

    if (index.compare("flood") == 0) {
        assert(idx_set.flood != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.flood), box_queries);
            return 0;
        }
    }

    if (index.compare("ifi") == 0) {
        assert(idx_set.ifi != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.ifi), box_queries);
            return 0;
        }
    }

    if (index.compare("fs") == 0) {
        assert(idx_set.fs != nullptr);
        if (mode.compare("range") == 0) {
            bench::query::custom_batch_range_queries(*(idx_set.fs), box_queries);
            return 0;
        }
        if (mode.compare("knn") == 0) {
            bench::query::custom_batch_knn_queries_var_k(*(idx_set.fs), knn_queries);
            return 0;
        }
    }
    
}
