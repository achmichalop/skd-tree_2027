#pragma once

#include <climits>
#include <bits/stdc++.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <queue>
#include <string>
#include <variant>
#include <immintrin.h>
#include <chrono>
#include "tree_core.hpp"
#include "heaps.hpp"


/*
 * SIMD-Based Node Search Utilities
 * -------------------------------
 * These helper functions implement dimension-specific search primitives for
 * internal nodes using AVX-512 intrinsics.
 *
 * The successorLinGNode{16,32,64} functions compute the successor position of a
 * query coordinate within a node by comparing the query value against all
 * stored separators in parallel. The result corresponds to the child index
 * that should be traversed during point queries.
 *
 * The intersectNode{16,32,64} functions compute the range of child positions
 * intersected by a query interval along a selected dimension. They return the
 * start and end child indices whose separator ranges overlap the query bounds
 * and are used to guide range queries.
 *
 * Separate implementations are provided for each internal node layout (N16,
 * N32, and N64) to match the corresponding separator representation and key
 * compression level. All comparisons are performed in a branch-free manner
 * using SIMD masks and population counts to maximize throughput.
 */

inline uint32_t successorLinGNode16(uint16_t *separators, uint64_t convertedCoordinate)
{
    __m512i threshold_vec = _mm512_set1_epi16((uint16_t) (convertedCoordinate >> NODE_16_TRAILING_ZEROS));
    __m512i separators_vec = _mm512_loadu_epi16((__m512i *) separators);
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu16_mask(threshold_vec, separators_vec);
    return _mm_popcnt_u32(maskResult); 
}
inline uint32_t successorLinGNode32(uint32_t *separators, uint64_t convertedCoordinate)
{
    __m512i threshold_vec = _mm512_set1_epi32((uint32_t) (convertedCoordinate >> NODE_32_TRAILING_ZEROS));
    __m512i separators_vec = _mm512_loadu_epi32((__m512i *) separators);
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu32_mask(threshold_vec, separators_vec);
    return _mm_popcnt_u32(maskResult);
}
inline uint32_t successorLinGNode64(uint64_t *separators, uint64_t convertedCoordinate)
{
    __m512i threshold_vec = _mm512_set1_epi64(convertedCoordinate);
    __m512i separators_vec = _mm512_loadu_epi64((__m512i *) separators);
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu64_mask(threshold_vec, separators_vec);
    return _mm_popcnt_u32(maskResult);
}

inline void intersectNode16(uint16_t *separators, uint64_t intervalStart, uint64_t intervalEnd, int &startPos, int &endPos)
{
    __m512i separators_vec = _mm512_loadu_epi16((__m512i *) separators);
    __m512i intervalStart_vec = _mm512_set1_epi16((uint16_t) (intervalStart >> NODE_16_TRAILING_ZEROS));
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu16_mask(intervalStart_vec, separators_vec);
    startPos = (int) _mm_popcnt_u32(maskResult);

    __m512i intervalEnd_vec = _mm512_set1_epi16((uint16_t) (intervalEnd >> NODE_16_TRAILING_ZEROS));
    maskResult = (uint32_t) _mm512_cmpge_epu16_mask(intervalEnd_vec, separators_vec);
    endPos = (int) _mm_popcnt_u32(maskResult);
}

inline void intersectNode32(uint32_t *separators, uint64_t intervalStart, uint64_t intervalEnd, int &startPos, int &endPos)
{
    __m512i separators_vec = _mm512_loadu_epi32((__m512i *) separators);
    __m512i intervalStart_vec = _mm512_set1_epi32((uint32_t) (intervalStart >> NODE_32_TRAILING_ZEROS));
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu32_mask(intervalStart_vec, separators_vec);
    startPos = (int) _mm_popcnt_u32(maskResult);

    __m512i intervalEnd_vec = _mm512_set1_epi32((uint32_t) (intervalEnd >> NODE_32_TRAILING_ZEROS));
    maskResult = (uint32_t) _mm512_cmpge_epu32_mask(intervalEnd_vec, separators_vec);
    endPos = (int) _mm_popcnt_u32(maskResult);
}

inline void intersectNode64(uint64_t *separators, uint64_t intervalStart, uint64_t intervalEnd, int &startPos, int &endPos)
{
    __m512i separators_vec = _mm512_loadu_epi64((__m512i *) separators);
    __m512i intervalStart_vec = _mm512_set1_epi64(intervalStart);
    uint32_t maskResult = (uint32_t) _mm512_cmpge_epu64_mask(intervalStart_vec, separators_vec);
    startPos = (int) _mm_popcnt_u32(maskResult);

    __m512i intervalEnd_vec = _mm512_set1_epi64(intervalEnd);
    maskResult = (uint32_t) _mm512_cmpge_epu64_mask(intervalEnd_vec, separators_vec);
    endPos = (int) _mm_popcnt_u32(maskResult);
}


/*
 * BFS Range Query Implementations
 * -------------------------------------
 * These functions evaluate axis-aligned range queries on skd-tree,
 * using a breadth-first traversal of internal nodes and SIMD-accelerated
 * filtering of leaf points.
 *
 * Behavior:
 * 1. Internal nodes (N16, N32, N64) are visited in BFS order. For each node,
 *    the `intersectNode{16,32,64}` helpers determine which child nodes
 *    intersect the query interval along the node's splitting dimension.
 * 2. Candidate children are pushed into the queue for further exploration.
 * 3. Leaf nodes are processed in two stages:
 *    - Full containment: if the leaf's bounding box is entirely within the
 *      query box, all points are added directly to the results.
 *    - Partial overlap: points are filtered using AVX-512 vectorized
 *      comparisons across dimensions, followed by scalar processing for
 *      leftover points.
 *
 * Layout Specializations:
 *  - rangeQuery_3way_nodes: supports N16, N32, and N64 internal nodes.
 *  - rangeQuery_2way_nodes: supports N32 and N64 internal nodes.
 *  - rangeQuery_1way_nodes: supports only N64 internal nodes.
 *
 * The `rangeQuery` table provides function pointers for dispatching queries
 * based on the selected tree configuration, avoiding runtime branching in
 * the hot query path.
 */


template <size_t Dim>
inline types::Points<Dim> rangeQuery_3way_nodes(types::Box<Dim> &box)
{   
    types::Points<Dim> results;
    void *node = nullptr;
    int startPos = 0, endPos = 0;
    uint8_t dim = 0, type;
    uint32_t counter = 0;
    std::queue<void*> queue;
    uint64_t convertedIntervalsStart[Dim];
    uint64_t convertedIntervalsEnd[Dim];
    types::Point<Dim> minCoords = box.min_corner(), maxCoords = box.max_corner();

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedIntervalsStart[i] = (uint64_t) (minCoords[i] * CONVERSION_FACTOR);
        convertedIntervalsEnd[i] = (uint64_t) (maxCoords[i] * CONVERSION_FACTOR);
    }

    queue.push(root);

    while (!queue.empty())
    {
        node = queue.front();
        queue.pop();
        uint32_t metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N16:
            {
                Node16 *actualNode = reinterpret_cast<Node16*>(node);
                intersectNode16(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }

                break;
            }
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32*>(node);
                intersectNode32(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64*>(node);
                intersectNode64(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = actualNode->slotuse;

                if (boost::geometry::covered_by(actualNode->boundingBox, box))
                {
                    int offset = results.size();
                    results.resize(offset + numRecords);
                    for (int i = 0; i < numRecords; i++)
                    {
                        types::Point<Dim> p;
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][i];
                        }
                        results[i + offset] = p;
                    }
                    break;
                }

                types::Point<Dim> leafMinCoords = actualNode->boundingBox.min_corner(), leafMaxCoords = actualNode->boundingBox.max_corner();

                bool hasResults = true;
                for (int j = 0; j < Dim; j++)
                {
                    if (leafMinCoords[j] > maxCoords[j] || leafMaxCoords[j] < minCoords[j])
                    {
                        hasResults = false;
                        break;
                    }
                }

                if (!hasResults)
                    break;
                
                uint32_t numBlocks = numRecords >> 3;   // N / 8 (8 64bit inside each 512-bit register)
                uint32_t activeBlocks = numBlocks;
                uint32_t remainder = numRecords & 7;    // N % 8 (those elements should be examined with scalar approach)
                std::vector<__mmask8> masks(numBlocks, 0xFF);

                for (int j = 0; j < Dim && activeBlocks; j++)
                {
                    if (leafMinCoords[j] >= minCoords[j] && leafMaxCoords[j] <= maxCoords[j])
                        continue;
                    
                    __m512d qLow_vec = _mm512_set1_pd(minCoords[j]);
                    __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);
                    
                    for (int b = 0; b < numBlocks; b++)
                    {

                        if (!masks[b])
                            continue;
                            
                        int offset = b << 3;
                        __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + offset);
                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        masks[b] &= (mask_le & mask_ge);

                        if(!masks[b])
                            activeBlocks--;
                    }
                }

                for (int b = 0; b < numBlocks; b++)
                {
                    __mmask8 mask = masks[b];

                    if (!mask)
                        continue;

                    int offset = b << 3;
                    while (mask)
                    {
                        uint16_t idx = _tzcnt_u16(mask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        mask &= (mask-1);
                    }
                }

                if (remainder)
                {
                    __mmask8 tailMask = (1u << remainder) - 1;
                    __mmask8 validMask = tailMask;

                    uint32_t baseIdx = numBlocks << 3;
                    for (int j = 0; j < Dim; j++)
                    {
                        __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, actualNode->records[j].data() + baseIdx);

                        __m512d qLow_vec  = _mm512_set1_pd(minCoords[j]);
                        __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);

                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        validMask &= (mask_le & mask_ge);

                        if (!validMask) break;
                    }
                
                    int offset = numBlocks << 3;
                    while (validMask)
                    {
                        uint16_t idx = _tzcnt_u16(validMask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        validMask &= (validMask-1);
                    }
                }
                break;
            }
        }
    }
    return results;
}

template <size_t Dim>
inline types::Points<Dim> rangeQuery_2way_nodes(types::Box<Dim> &box)
{   
    types::Points<Dim> results;
    void *node = nullptr;
    int startPos = 0, endPos = 0;
    uint8_t dim = 0, type;
    uint32_t counter = 0;
    std::queue<void*> queue;
    uint64_t convertedIntervalsStart[Dim];
    uint64_t convertedIntervalsEnd[Dim];
    types::Point<Dim> minCoords = box.min_corner(), maxCoords = box.max_corner();

    for (uint32_t i = 0; i < Dim; i++)
    {

        convertedIntervalsStart[i] = (uint64_t) (minCoords[i] * CONVERSION_FACTOR);
        convertedIntervalsEnd[i] = (uint64_t) (maxCoords[i] * CONVERSION_FACTOR);
    }

    queue.push(root);
    while (!queue.empty())
    {
        node = queue.front();
        queue.pop();
        uint32_t metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32*>(node);
                intersectNode32(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64*>(node);
                intersectNode64(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = actualNode->slotuse;
                if (boost::geometry::covered_by(actualNode->boundingBox, box))
                {
                    int offset = results.size();
                    results.resize(offset + numRecords);
                    for (int i = 0; i < numRecords; i++)
                    {
                        types::Point<Dim> p;
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][i];
                        }
                        results[i + offset] = p;
                    }

                    break;
                }

                types::Point<Dim> leafMinCoords = actualNode->boundingBox.min_corner(), leafMaxCoords = actualNode->boundingBox.max_corner();

                bool hasResults = true;
                for (int j = 0; j < Dim; j++)
                {
                    if (leafMinCoords[j] > maxCoords[j] || leafMaxCoords[j] < minCoords[j])
                    {
                        hasResults = false;
                        break;
                    }
                }
                
                if (!hasResults)
                {
                    break;
                }

                
                uint32_t numBlocks = numRecords >> 3;   // N / 8 (8 64bit inside each 512-bit register)
                uint32_t activeBlocks = numBlocks;
                uint32_t remainder = numRecords & 7;    // N % 8 (those elements should be examined with scalar approach)
                std::vector<__mmask8> masks(numBlocks, 0xFF);
                
                for (int j = 0; j < Dim && activeBlocks; j++)
                {
                    if (leafMinCoords[j] >= minCoords[j] && leafMaxCoords[j] <= maxCoords[j])
                        continue;
                    
                    __m512d qLow_vec = _mm512_set1_pd(minCoords[j]);
                    __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);
                    
                    for (int b = 0; b < numBlocks; b++)
                    {

                        if (!masks[b])
                            continue;
                            
                        int offset = b << 3;
                        __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + offset);
                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        masks[b] &= (mask_le & mask_ge);

                        if(!masks[b])
                            activeBlocks--;
                    }
                }

                for (int b = 0; b < numBlocks; b++)
                {
                    __mmask8 mask = masks[b];

                    if (!mask)
                        continue;

                    int offset = b << 3;
                    while (mask)
                    {
                        uint16_t idx = _tzcnt_u16(mask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        mask &= (mask-1);
                    }
                }

                if (remainder)
                {
                    __mmask8 tailMask = (1u << remainder) - 1;
                    __mmask8 validMask = tailMask;

                    uint32_t baseIdx = numBlocks << 3;
                    for (int j = 0; j < Dim; j++)
                    {
                        __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, actualNode->records[j].data() + baseIdx);

                        __m512d qLow_vec  = _mm512_set1_pd(minCoords[j]);
                        __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);

                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        validMask &= (mask_le & mask_ge);

                        if (!validMask) break;
                    }
                
                    int offset = numBlocks << 3;
                    while (validMask)
                    {
                        uint16_t idx = _tzcnt_u16(validMask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        validMask &= (validMask-1);
                    }
                }

                break;
            }
        }
    }
    
    return results;

}

template <size_t Dim>
inline types::Points<Dim> rangeQuery_1way_nodes(types::Box<Dim> &box)
{   
    types::Points<Dim> results;
    void *node = nullptr;
    int startPos = 0, endPos = 0;
    uint8_t dim = 0, type;
    uint32_t counter = 0;
    std::queue<void*> queue;
    uint64_t convertedIntervalsStart[Dim];
    uint64_t convertedIntervalsEnd[Dim];
    types::Point<Dim> minCoords = box.min_corner(), maxCoords = box.max_corner();

    for (uint32_t i = 0; i < Dim; i++)
    {

        convertedIntervalsStart[i] = (uint64_t) (minCoords[i] * CONVERSION_FACTOR);
        convertedIntervalsEnd[i] = (uint64_t) (maxCoords[i] * CONVERSION_FACTOR);
    }

    queue.push(root);

    while (!queue.empty())
    {
        node = queue.front();
        queue.pop();
        uint32_t metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64*>(node);
                intersectNode64(actualNode->separators, convertedIntervalsStart[dim], convertedIntervalsEnd[dim], startPos, endPos);
                for (int i = startPos; i <= endPos; i++)
                {
                    queue.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = actualNode->slotuse;

                if (boost::geometry::covered_by(actualNode->boundingBox, box))
                {
                    int offset = results.size();
                    results.resize(offset + numRecords);
                    for (int i = 0; i < numRecords; i++)
                    {
                        types::Point<Dim> p;
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][i];
                        }
                        results[i + offset] = p;
                    }
                    break;
                }

                types::Point<Dim> leafMinCoords = actualNode->boundingBox.min_corner(), leafMaxCoords = actualNode->boundingBox.max_corner();

                bool hasResults = true;
                for (int j = 0; j < Dim; j++)
                {
                    if (leafMinCoords[j] > maxCoords[j] || leafMaxCoords[j] < minCoords[j])
                    {
                        hasResults = false;
                        break;
                    }
                }
                
                if (!hasResults)
                    break;
                
                uint32_t numBlocks = numRecords >> 3;   // N / 8 (8 64bit inside each 512-bit register)
                uint32_t activeBlocks = numBlocks;
                uint32_t remainder = numRecords & 7;    // N % 8 (those elements should be examined with scalar approach)
                std::vector<__mmask8> masks(numBlocks, 0xFF);

                for (int j = 0; j < Dim && activeBlocks; j++)
                {
                    if (leafMinCoords[j] >= minCoords[j] && leafMaxCoords[j] <= maxCoords[j])
                        continue;
                    
                    __m512d qLow_vec = _mm512_set1_pd(minCoords[j]);
                    __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);
                    
                    for (int b = 0; b < numBlocks; b++)
                    {

                        if (!masks[b])
                            continue;
                            
                        int offset = b << 3;
                        __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + offset);
                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        masks[b] &= (mask_le & mask_ge);

                        if(!masks[b])
                            activeBlocks--;
                    }
                }

                for (int b = 0; b < numBlocks; b++)
                {
                    __mmask8 mask = masks[b];

                    if (!mask)
                        continue;

                    int offset = b << 3;
                    while (mask)
                    {
                        uint16_t idx = _tzcnt_u16(mask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        mask &= (mask-1);
                    }
                }

                if (remainder)
                {
                    __mmask8 tailMask = (1u << remainder) - 1;
                    __mmask8 validMask = tailMask;

                    uint32_t baseIdx = numBlocks << 3;
                    for (int j = 0; j < Dim; j++)
                    {
                        __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, actualNode->records[j].data() + baseIdx);

                        __m512d qLow_vec  = _mm512_set1_pd(minCoords[j]);
                        __m512d qHigh_vec = _mm512_set1_pd(maxCoords[j]);

                        __mmask8 mask_le = _mm512_cmp_pd_mask(coords_vec, qLow_vec, _CMP_GE_OQ);
                        __mmask8 mask_ge = _mm512_cmp_pd_mask(coords_vec, qHigh_vec, _CMP_LE_OQ);
                        validMask &= (mask_le & mask_ge);

                        if (!validMask) break;
                    }
                
                    int offset = numBlocks << 3;
                    while (validMask)
                    {
                        uint16_t idx = _tzcnt_u16(validMask);
                        results.emplace_back();
                        auto &p = results.back();
                        for (int j = 0; j < Dim; j++)
                        {
                            p[j] = actualNode->records[j][offset + idx];
                        }

                        validMask &= (validMask-1);
                    }
                }
                break;
            }
        }
    }
    
    return results;

}


template <size_t Dim>
using RangeQuery_func = types::Points<Dim>(*)(types::Box<Dim> &);

template <size_t Dim>
inline RangeQuery_func<Dim> rangeQuery[] = {
    rangeQuery_3way_nodes<Dim>,
    rangeQuery_2way_nodes<Dim>, 
    rangeQuery_1way_nodes<Dim> 
};


/*
 * K-Nearest Neighbor (KNN) Query Implementations
 * ----------------------------------------------
 * These functions perform k-nearest neighbor queries on skd-tree instances
 * using a best-first search strategy implemented with a min-heap.
 *
 * Key Features:
 * 1. **Heap-based traversal**: Internal nodes are explored in increasing
 *    order of projected distance from the query point.
 * 2. **Successor selection in internal nodes**: For N16, N32, and N64 nodes,
 *    the `successorLinGNode{16,32,64}` helpers find the child whose
 *    splitting plane is closest to the query coordinate along the node's
 *    splitting dimension.
 * 3. **Left/Right Group propagation**: Nodes left or right of the closest
 *    child are added to the heap with their estimated distances (squared
 *    projections) to allow pruning based on the current k-nearest candidates.
 * 4. **Leaf processing**:
 *    - Uses AVX-512 vectorization to compute squared distances between
 *      query and leaf points across all dimensions.
 *    - Scalar fallback handles points not divisible by the vector width.
 *    - Candidate points are stored in a fixed-size max-heap; the farthest
 *      current candidate defines a `bound` to prune further exploration.
 *
 * Layout Specializations:
 *  - knnQuery_3way_nodes: supports N16, N32, N64 internal nodes.
 *  - knnQuery_2way_nodes: supports N32, N64 internal nodes.
 *  - knnQuery_1way_nodes: supports only N64 internal nodes.
 *
 * The `knnQuery` table provides function pointers for dispatching queries
 * based on the tree layout to avoid runtime branching in hot paths.
 */

template <size_t Dim>
inline types::Points<Dim> knnQuery_3way_nodes(types::Point<Dim> query, uint32_t k)
{
    heaps::minHeap<Dim> heap;
    heaps::minHeapNode<Dim> heapEntry;
    heaps::maxHeap<Dim> candidateResults(k);
    types::Points<Dim> results;
    uint8_t dim = 0, type;
    uint32_t metadata;
    double bound = std::numeric_limits<double>::max();
    uint64_t convertedCoordinates[Dim];
    
    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (query[i] * CONVERSION_FACTOR);
    }

    heap.push(heaps::minHeapNode<Dim>(root, 0, heaps::SINGLE_NODE, 0.0f, std::array<double, Dim>{}));

    while(!heap.empty())
    {
        heapEntry = heap.top();
        heap.pop();

        if (candidateResults.currCapacity == k && heapEntry.dist >= bound)
        {
            for (auto& entry: candidateResults)
            {
                results.emplace_back(entry.p);
            }

            break;
        }

        switch (heapEntry.type)
        {
            case heaps::SINGLE_NODE:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N16:
                    {

                        Node16* actualNode = reinterpret_cast<Node16 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode16(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_16_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_16_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                        
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }

                        break;
                    }
                }

                break;
            }
            case heaps::LEFT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;
    
                switch (type)
                {
                    case NodeType::N16:
                    {
                        Node16* actualNode = reinterpret_cast<Node16 *>(heapEntry.node);
                        if(heapEntry.idx > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx - 1;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos] << NODE_16_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);
                        if(heapEntry.idx > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx - 1;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);
                        if(heapEntry.idx > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx - 1;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }                
                }
                
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N16:
                    {

                        Node16* actualNode = reinterpret_cast<Node16 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode16(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_16_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_16_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;
                            
                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                        
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }
                    
                        break;
                    }
                }

                break;
            }
            case heaps::RIGHT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;
    
                switch (type)
                {
                    case NodeType::N16:
                    {
                        Node16* actualNode = reinterpret_cast<Node16 *>(heapEntry.node);
                        if(heapEntry.idx < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx + 1;
                            double diff = (((uint64_t) actualNode->separators[pos-1] << NODE_16_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);
                        if(heapEntry.idx < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx + 1;
                            double diff = (((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);
                        if(heapEntry.idx < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx + 1;
                            double diff = (actualNode->separators[pos-1] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }                
                }
                
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N16:
                    {

                        Node16* actualNode = reinterpret_cast<Node16 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode16(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_16_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_16_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;
                            
                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                    
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }

                        break;
                    }
                }

                break;
            }
        }

    }
    
    return results;
}

template <size_t Dim>
inline types::Points<Dim> knnQuery_2way_nodes(types::Point<Dim> query, uint32_t k)
{
    heaps::minHeap<Dim> heap;
    heaps::minHeapNode<Dim> heapEntry;
    heaps::maxHeap<Dim> candidateResults(k);
    types::Points<Dim> results;
    uint8_t dim = 0, type;
    uint32_t metadata;
    double bound = std::numeric_limits<double>::max();
    uint64_t convertedCoordinates[Dim];
    
    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (query[i] * CONVERSION_FACTOR);
    }

    heap.push(heaps::minHeapNode<Dim>(root, 0, heaps::SINGLE_NODE, 0.0f, std::array<double, Dim>{}));

    while(!heap.empty())
    {
        heapEntry = heap.top();
        heap.pop();

        if (candidateResults.currCapacity == k && heapEntry.dist >= bound)
        {
            for (auto& entry: candidateResults)
            {
                results.emplace_back(entry.p);
            }

            break;
        }

        switch (heapEntry.type)
        {
            case heaps::SINGLE_NODE:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }
                        
                        break;
                    }
                }

                break;
            }
            case heaps::LEFT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;
    
                switch (type)
                {
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);
                        if(heapEntry.idx > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx - 1;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);
                        if(heapEntry.idx > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx - 1;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }                
                }
                
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                     
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }

                        break;
                    }
                }

                break;
            }
            case heaps::RIGHT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;
    
                switch (type)
                {
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);
                        if(heapEntry.idx < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx + 1;
                            double diff = (((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);
                        if(heapEntry.idx < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            uint32_t pos = heapEntry.idx + 1;
                            double diff = (actualNode->separators[pos-1] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                        }

                        heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                        heapEntry.type = heaps::SINGLE_NODE;
                        break;
                    }                
                }
                
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N32:
                    {
                        Node32* actualNode = reinterpret_cast<Node32 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - ((uint64_t) actualNode->separators[pos-1] << NODE_32_TRAILING_ZEROS)) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (((uint64_t) actualNode->separators[pos] << NODE_32_TRAILING_ZEROS) - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }

                        break;
                    }
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;
                        
                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }
                        
                        break;
                    }
                }

                break;
            }
        }

    }
    
    return results;
}

template <size_t Dim>
inline types::Points<Dim> knnQuery_1way_nodes(types::Point<Dim> query, uint32_t k)
{
    heaps::minHeap<Dim> heap;
    heaps::minHeapNode<Dim> heapEntry;
    heaps::maxHeap<Dim> candidateResults(k);
    types::Points<Dim> results;
    uint8_t dim = 0, type;
    uint32_t metadata;
    double bound = std::numeric_limits<double>::max();
    uint64_t convertedCoordinates[Dim];
    
    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (query[i] * CONVERSION_FACTOR);
    }

    heap.push(heaps::minHeapNode<Dim>(root, 0, heaps::SINGLE_NODE, 0.0f, std::array<double, Dim>{}));

    while(!heap.empty())
    {
        heapEntry = heap.top();
        heap.pop();

        if (candidateResults.currCapacity == k && heapEntry.dist >= bound)
        {
            for (auto& entry: candidateResults)
            {
                results.emplace_back(entry.p);
            }

            break;
        }

        switch (heapEntry.type)
        {
            case heaps::SINGLE_NODE:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;

                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }

                        break;
                    }
                }

                break;
            }
            case heaps::LEFT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;
                
                Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);
                if(heapEntry.idx > 0)
                {
                    std::array<double, Dim> projections = heapEntry.projection_dists;
                    uint32_t pos = heapEntry.idx - 1;
                    double diff = (convertedCoordinates[dim] - actualNode->separators[pos]) / (double) CONVERSION_FACTOR;
                    double projection = diff * diff - projections[dim];
                    double dist = heapEntry.dist + projection;
                    projections[dim] = projection;

                    if (dist < bound)
                        heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::LEFT_GROUP, dist, projections));
                }

                heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                heapEntry.type = heaps::SINGLE_NODE;
                
                metadata = *reinterpret_cast<uint16_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;

                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }
                        break;
                    }
                }

                break;
            }
            case heaps::RIGHT_GROUP:
            {
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                if(heapEntry.idx < actualNode->slotuse - 1)
                {
                    std::array<double, Dim> projections = heapEntry.projection_dists;
                    uint32_t pos = heapEntry.idx + 1;
                    double diff = (actualNode->separators[pos-1] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                    double projection = diff * diff - projections[dim];
                    double dist = heapEntry.dist + projection;
                    projections[dim] = projection;

                    if (dist < bound)
                        heap.push(heaps::minHeapNode<Dim>(actualNode, pos, heaps::RIGHT_GROUP, dist, projections));
                }

                heapEntry.node = actualNode->childsPtrs[heapEntry.idx];
                heapEntry.type = heaps::SINGLE_NODE;
                
                metadata = *reinterpret_cast<uint32_t *>(heapEntry.node);
                type = metadata & 0x3;
                dim = (metadata >> 2) & 0x3F;

                switch (type)
                {
                    case NodeType::N64:
                    {
                        Node64* actualNode = reinterpret_cast<Node64 *>(heapEntry.node);

                        uint32_t pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                        heap.push(heaps::minHeapNode<Dim>(actualNode->childsPtrs[pos], pos, heaps::SINGLE_NODE, heapEntry.dist, heapEntry.projection_dists));

                        if (pos > 0)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (convertedCoordinates[dim] - actualNode->separators[pos-1]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos-1, heaps::LEFT_GROUP, dist, projections));
                        }

                        if (pos < actualNode->slotuse - 1)
                        {
                            std::array<double, Dim> projections = heapEntry.projection_dists;
                            double diff = (actualNode->separators[pos] - convertedCoordinates[dim]) / (double) CONVERSION_FACTOR;
                            double projection = diff * diff - projections[dim];
                            double dist = heapEntry.dist + projection;
                            projections[dim] = projection;

                            if (dist < bound)
                                heap.push(heaps::minHeapNode<Dim>(actualNode, pos + 1, heaps::RIGHT_GROUP, dist, projections));
                        }
                        break;
                    }
                    case NodeType::Leaf:
                    {
                        LeafNode<Dim> *actualNode = reinterpret_cast<LeafNode<Dim>*>(heapEntry.node);

                        uint32_t numOfRecords = actualNode->slotuse;

                        std::vector<double> dists(numOfRecords, 0.0f);

                        __m512d q_vec[Dim];
                        for (uint32_t j = 0; j < Dim; j++)
                        {
                            q_vec[j] = _mm512_set1_pd(query[j]);
                        }


                        uint32_t i = 0;
                        for (; i + 8 < numOfRecords; i += 8)
                        {
                            __m512d dist_vec = _mm512_setzero_pd();

                            for (uint32_t j = 0; j < Dim; j++)
                            { 
                                __m512d coords_vec = _mm512_loadu_pd(actualNode->records[j].data() + i);
                                __m512d diff_vec = _mm512_sub_pd(coords_vec, q_vec[j]);
                                dist_vec = _mm512_fmadd_pd(diff_vec, diff_vec, dist_vec);
                            }
                            _mm512_storeu_pd(dists.data() + i, dist_vec);
                        }

                        for (; i < numOfRecords; i++)
                        {
                            double dist = 0.0f;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                double diff = actualNode->records[j][i] - query[j];
                                dist += diff * diff;
                            }
                            dists[i] = dist;
                        }

                        for (i = 0; i < numOfRecords; i++)
                        {
                            types::Point<Dim> p;
                            for (uint32_t j = 0; j < Dim; j++)
                            {
                                p[j] = actualNode->records[j][i];
                            }

                            candidateResults.insert(heaps::maxHeapNode(dists[i], p));
                        }

                        if (candidateResults.size() == candidateResults.capacity)
                        {
                            bound = candidateResults.at(0).dist;
                        }
                        
                        break;
                    }
                }

                break;
            }
        }

    }
    
    return results;
}

template <size_t Dim>
using knnQuery_func = types::Points<Dim>(*)(types::Point<Dim>, uint32_t);

template <size_t Dim>
inline knnQuery_func<Dim> knnQuery[] = {
    knnQuery_3way_nodes<Dim>,
    knnQuery_2way_nodes<Dim>, 
    knnQuery_1way_nodes<Dim> 
};

