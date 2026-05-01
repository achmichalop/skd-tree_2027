#pragma once

#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include "../../../utils/type.hpp"
#include "../../../utils/common.hpp"



/*
 * tree_core.hpp
 * -------------
 * This file defines the core data structures and memory management utilities
 * used by the skd-tree index.
 *
 * It contains the declarations of all node layouts used by the tree, including
 * three types of internal nodes (N16, N32, and N64) and a template-based leaf
 * node layout.
 *
 * The internal node layouts correspond to different levels of key compression:
 *   - N16 and N32 implement compressed internal node representations,
 *     reducing memory footprint by storing prefixes of separators (i.e., splitters)
 *     using smaller integer types.
 *   - N64 represents the baseline, uncompressed layout using 64-bit (i.e., exact)
 *     separators.
 *
 * The leaf node layout is dimension-dependent and follows a Structure-of-Arrays
 * (SoA) organization for storing point coordinates. Each leaf additionally
 * maintains a bounding box in the D-dimensional space to support pruning and
 * efficient query processing.
 *
 * This file also defines two dedicated node pool allocators responsible for
 * managing the memory layout of internal nodes and leaf nodes, respectively.
 * These pools allocate nodes in large, cache-aligned blocks to minimize
 * allocation overhead and improve locality.
 *
 * Finally, tree_core.hpp provides the logic for selecting the appropriate tree
 * configuration at construction time. Based on the expected number of leaves,
 * dimensionality, and split characteristics, it determines whether internal
 * node compression should be applied and which node layout (N16, N32, or N64)
 * is enabled in order to form hyper-cubic leaf partitions.
 */


namespace types
{
    template <size_t Dim>
    using Point = point_t<Dim>;
    template <size_t Dim>
    using Points = std::vector<Point<Dim>>;
    template <size_t Dim>
    using Box = box_t<Dim>;
}


#define NODE_SIZE_16 32
#define NODE_SIZE_32 16
#define NODE_SIZE_64 8
#define NODE_DEPTH_16 5
#define NODE_DEPTH_32 4
#define NODE_DEPTH_64 3
#define NODE_16_TRAILING_ZEROS    48
#define NODE_32_TRAILING_ZEROS    32
#define CONVERSION_FACTOR (uint64_t) ((ULONG_MAX - 1) >> 1)

enum NodeType {N16 = 0, N32, N64 , Leaf}; 
enum TreeSelection {three_way_internal_nodes = 0, two_way_internal_nodes, one_way_internal_nodes};

typedef struct alignas(64) Node_16bit
{
    uint32_t type : 2;
    uint32_t dim : 6;
    uint32_t slotuse : 24;
    uint16_t separators[NODE_SIZE_16];
    void* childsPtrs[NODE_SIZE_16];      
}Node16;

typedef struct alignas(64) Node_32bit
{
    uint32_t type : 2;
    uint32_t dim : 6;
    uint32_t slotuse : 24;                    
    uint32_t separators[NODE_SIZE_32];
    void* childsPtrs[NODE_SIZE_32];
}Node32;

typedef struct alignas(64) Node_64bit
{
    uint32_t type : 2;
    uint32_t dim : 6;
    uint32_t slotuse : 24;                      
    uint64_t separators[NODE_SIZE_64];
    void* childsPtrs[NODE_SIZE_64];
}Node64;

template <size_t Dim>
struct Leaf_Node
{
    uint32_t type : 2;
    uint32_t dim : 6;
    uint32_t slotuse : 23;
    uint32_t is_outlier : 1;
    types::Box<Dim> boundingBox;
    std::array<std::vector<double>, Dim> records;
};

template <size_t Dim>
using LeafNode = Leaf_Node<Dim>;


class NodePool {
private:
    size_t blockSize;
    size_t currentIndex;
    size_t maxNodeSize;
    std::vector<void*> blocks;

    NodePool(size_t largestNodeSize, size_t blockSize)
        : maxNodeSize(largestNodeSize), blockSize(blockSize), currentIndex(blockSize) {}

public:

    size_t get_block_count() const { return blocks.size(); }
    size_t get_block_size_bytes() const { return maxNodeSize * blockSize; }
    size_t get_node_size() const { return maxNodeSize; }
    size_t get_used_nodes() const {
        return (blocks.size() - 1) * blockSize + currentIndex;
    }
    const std::vector<void*>& get_blocks() const { return blocks; }
    
    static NodePool* Create(size_t largestNodeSize, size_t blockSize) {
        return new NodePool(largestNodeSize, blockSize);
    }

    void* allocate() {
        if(currentIndex >= blockSize) allocateBlock();
        void* ptr = static_cast<char*>(blocks.back()) + currentIndex * maxNodeSize;
        ++currentIndex;
        return ptr;
    }

    void clear() {
        for(auto block : blocks) free(block);
        blocks.clear();
        currentIndex = blockSize;
    }

private:
    void allocateBlock() {
        void* ptr = nullptr;
        if(posix_memalign(&ptr, 64, maxNodeSize * blockSize) != 0)
            throw std::bad_alloc();
        blocks.push_back(ptr);
        currentIndex = 0;
    }
};


struct tree_statistics
{
    size_t treeDepth = 0;
    size_t outlierThreshold = 0;
    size_t splitThreshold = 0;
    size_t maxLeafCapacity = 0;
};

inline tree_statistics treeStats;
inline void *root;
inline size_t leafCapacity = 128;
inline size_t expectedNumOfLeaves = 0;
inline size_t expectedSplitsPerDimension = 0;
inline TreeSelection treeType;
inline NodePool* internalNodePool = nullptr;
inline NodePool* leafNodePool = nullptr;

template <size_t Dim>
inline void initialize_node_pools()
{
    size_t largestNodeSize = 0;

    switch (treeType)
    {
        case TreeSelection::three_way_internal_nodes:
        {
            largestNodeSize = sizeof(Node16);
            break;
        } 
        case TreeSelection::two_way_internal_nodes:
        {
            largestNodeSize = sizeof(Node32);
            break;
        }
        case TreeSelection::one_way_internal_nodes:
        {
            largestNodeSize = sizeof(Node64);
            break;
        }
    }

    internalNodePool = NodePool::Create(largestNodeSize, expectedNumOfLeaves);
    leafNodePool = NodePool::Create(sizeof(LeafNode<Dim>), 2 * expectedNumOfLeaves);
}

template <size_t Dim>
inline void select_tree_type(size_t numOfPoints)
{
    expectedNumOfLeaves = (numOfPoints + leafCapacity - 1) / leafCapacity;
    expectedSplitsPerDimension = std::ceil(std::pow(expectedNumOfLeaves, 1.0f/Dim));

    if (expectedSplitsPerDimension > NODE_SIZE_32)
        treeType =  TreeSelection::three_way_internal_nodes;
    else if (expectedSplitsPerDimension > NODE_SIZE_64)
        treeType = TreeSelection::two_way_internal_nodes;
    else
        treeType = TreeSelection::one_way_internal_nodes;

    initialize_node_pools<Dim>();
}




