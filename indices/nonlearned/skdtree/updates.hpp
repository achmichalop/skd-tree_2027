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
#include <unordered_map>
#include "tree_core.hpp"
#include "heaps.hpp"
#include "evaluation.hpp"


/*
 * skd-tree Insertion Functions
 * ----------------------------
 * These routines insert a point into an skd-tree, supporting different
 * internal node configurations:
 * 
 * 1. insert_3way_nodes   → N16, N32, N64 internal nodes allowed
 * 2. insert_2way_nodes   → N32, N64 internal nodes allowed
 * 3. insert_1way_nodes   → only N64 internal nodes allowed
 *
 * Insertion Procedure:
 * -------------------
* 1. **Tree Traversal**:
 *    - Traverse from the root down to the leaf corresponding to the
 *      converted coordinates.
 *    - At each level, the appropriate successor function
 *      (`successorLinGNode16/32/64`) finds the child node closest to the
 *      query along the current split dimension.
 *    - The path is recorded along with a boolean indicating whether the
 *      node has space for new children (used for potential splits).
 *
 * 2. **Leaf Insertion**:
 *    - If the leaf has space (`slotuse < splitThreshold`), the point is
 *      inserted and the bounding box of the leaf is updated.
 *    - If the leaf is near capacity but below the outlier threshold,
 *      the point is appended, and the bounding box is updated.
 *    - If the leaf is an outlier and has capacity for extra points,
 *      it is appended without immediate splitting.
 *
 * 3. **Splitting Logic**:
 *    - If the leaf is full and there is a parent with space, the appropriate
 *      `make_split_{1/2/3}way_nodes` function is called to split the leaf
 *      and propagate new separators up.
 *    - If no immediate split is possible, a higher ancestor with available
 *      slots is found and `reconstruct_subtree_{1/2/3}way_nodes` is called.
 *    - If no ancestor has space, the point is added as an outlier, and
 *      the leaf’s bounding box is updated.
 *
 * The `insertPoint` table provides function pointers for dispatching insertions
 * based on the selected tree configuration, avoiding runtime branching in
 * the hot insertion path.
 * 
 * Notes:
 *  - All three insertion routines share a common path traversal and leaf
 *    handling logic; only the internal node types differ.
 *  - Bounding boxes are always updated to maintain correctness of future queries.
 *  - Outlier nodes allow temporary over-capacity points without triggering
 *    immediate tree restructuring.
 */


template<size_t Dim>
inline void update_subtree_3way_nodes(void *root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t>& dimPerLvl)
{  
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector<uint32_t> bounds_node16(NODE_SIZE_16, 0), bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node16(NODE_SIZE_16, ULONG_MAX), separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void *> currLvlNodes, nextLvlNodes;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    currLvlNodes.emplace_back(root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level; l < treeStats.treeDepth; l++)
    {
        numNextLvlNodes = 0;

        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else 
        {
            exponent = treeStats.treeDepth - l;
        }

        for (j = 0; j < numNodesPerLevel; j++)
        {
            void *node = currLvlNodes[j];
            metadata = *reinterpret_cast<uint32_t *>(node);
            nodeType = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;

            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});

            uint32_t subspaceLeaves = round(numOfElements/(double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));

            switch (nodeType)
            {
                case NodeType::N16:
                {
                    Node16 *actualNode = reinterpret_cast<Node16 *>(node);

                    if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
                    {
                        actualNode->slotuse = 1;
                        actualNode->separators[0] = USHRT_MAX;
                        nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                        nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                        numNextLvlNodes++;

                        for (i = actualNode->slotuse; i < NODE_SIZE_16; i++)
                        {
                            actualNode->separators[i] = USHRT_MAX;
                        }

                        break;
                    }

                    uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
                    extractSeparatorsNode16(points, dim, separators_node16, expectedSeparators, bounds_node16, startBound, endBound);
                    uint32_t producedSeparators = NODE_SIZE_16 - count(separators_node16.begin(), separators_node16.end(), 0);
                
                    actualNode->slotuse = producedSeparators;

                    for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_16; i++)
                    {
                        if (separators_node16[i])
                        {
                            actualNode->separators[k] = (uint16_t) (separators_node16[i] >> NODE_16_TRAILING_ZEROS);
                            nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                            nextLvlBounds.emplace_back(make_pair(startBound, bounds_node16[i]));
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node16[i];
                        }
                        else
                        {
                            actualNode->separators[g] = USHRT_MAX;
                            g++;
                        }
                    }
                
                    fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                    fill(bounds_node16.begin(), bounds_node16.end(), 0);

                    break;
                }
                case NodeType::N32:
                {
                    Node32 *actualNode = reinterpret_cast<Node32 *>(node);

                    if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
                    {
                        actualNode->slotuse = 1;
                        actualNode->separators[0] = UINT_MAX;
                        nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                        nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                        numNextLvlNodes++;

                        for (i = actualNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            actualNode->separators[i] = UINT_MAX;
                        }

                        break;
                    }
                    
                    uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
                    extractSeparatorsNode32(points, dim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    uint32_t producedSeparators = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    actualNode->slotuse = producedSeparators;
                
                    for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_32; i++)
                    {
                        if (separators_node32[i])
                        {
                            actualNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                            nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                            nextLvlBounds.emplace_back(make_pair(startBound, bounds_node32[i]));
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node32[i];
                        }
                        else
                        {
                            actualNode->separators[g] = UINT_MAX;
                            g++;
                        }
                    }

                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);

                    break;
                }
                case NodeType::N64:
                {
                    Node64 *actualNode = reinterpret_cast<Node64 *>(node);

                    if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
                    {
                        actualNode->slotuse = 1;
                        actualNode->separators[0] = ULONG_MAX;
                        nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                        nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                        numNextLvlNodes++;

                        for (i = actualNode->slotuse; i < NODE_SIZE_64; i++)
                        {
                            actualNode->separators[i] = ULONG_MAX;
                        }

                        break;
                    }

                    uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
                    extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    uint32_t producedSeparators = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    actualNode->slotuse = producedSeparators;
                
                    for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            actualNode->separators[k] = separators_node64[i];
                            nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                            nextLvlBounds.emplace_back(make_pair(startBound, bounds_node64[i]));
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node64[i];
                        }
                        else
                        {
                            actualNode->separators[g] = ULONG_MAX;
                            g++;
                        }
                    }

                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);
                    
                    break;
                }
            }
        }

        currLvlNodes = move(nextLvlNodes);
        nextLvlNodes.clear();
        
        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;

        dimPerLvl.emplace_back(dim);

    }

    for (j = 0; j < numNodesPerLevel; j++)
    {

        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(currLvlNodes[j]);
        dim = currLeaf->dim;


        if (numOfElements <= treeStats.splitThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            currLeaf->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            currLeaf->records[i].resize(leafNodeSize);
        }

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];

            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j][k] = p[j];
            }
        }

        currLeaf->slotuse = numOfElements;
        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);
    }
}

template<size_t Dim>
inline void create_subtree_3way_nodes(void *&root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t> dimPerLvl)
{
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node16(NODE_SIZE_16, 0), bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node16(NODE_SIZE_16, ULONG_MAX), separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    prevLvlPtrs.emplace_back(&root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level, d = 0; l < treeStats.treeDepth; l++, d++)
    {
        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else
        {
            exponent = treeStats.treeDepth - l;
        }

        numNextLvlNodes = 0;
        currLvlPtrs.resize(NODE_SIZE_16 * numNodesPerLevel);
        nextLvlBounds.resize(NODE_SIZE_16 * numNodesPerLevel);

        dim = dimPerLvl[d];

        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});
            uint32_t subspaceLeaves = round(numOfElements / (double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));

            if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
            {
                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                newNode->dim = dim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, endBound);
                numNextLvlNodes++;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }

                *prevLvlPtrs[j] = (void *) newNode;

                continue;
            }

            if (localSplits > NODE_SIZE_32)
                nodeType = N16;
            else if (localSplits > NODE_SIZE_64)
                nodeType = N32;
            else
                nodeType = N64;

            switch (nodeType)
            {
                case N16:
                {
                    uint32_t expectedSeparators = (localSplits >= NODE_SIZE_16) ? NODE_SIZE_16 : localSplits;
                    extractSeparatorsNode16(points, dim, separators_node16, expectedSeparators, bounds_node16, startBound, endBound);
                    producedSeparators[0] = NODE_SIZE_16 - count(separators_node16.begin(), separators_node16.end(), 0);
                    
                    if (producedSeparators[0] >= NODE_SIZE_32)
                    {
                        Node16 *newNode = static_cast<Node16 *>(internalNodePool->allocate());
                        newNode->dim = dim;
                        newNode->type = N16;
                        newNode->slotuse = producedSeparators[0];

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_16; i++)
                        {
                            if (separators_node16[i])
                            {
                                newNode->separators[k] = (uint16_t) (separators_node16[i] >> NODE_16_TRAILING_ZEROS);
                                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node16[i]);
                                numNextLvlNodes++;
                                k++;
                                startBound = bounds_node16[i];
                            }
                            else
                            {
                                newNode->separators[g] = USHRT_MAX;
                                newNode->childsPtrs[g] = nullptr;
                                g++;
                            }
                        }

                        *prevLvlPtrs[j] = (void *) newNode;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                        fill(bounds_node16.begin(), bounds_node16.end(), 0);
                        break;
                    }
                    else
                    {
                        expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                        extractSeparatorsNode32(points, dim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                        producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                        if (producedSeparators[1] > producedSeparators[0])
                        {
                            Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                            newNode->dim = dim;
                            newNode->type = N32;
                            newNode->slotuse = producedSeparators[1];

                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                            {
                                if (separators_node32[i])
                                {
                                    newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                    currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                    numNextLvlNodes++;
                                    k++;
                                    startBound = bounds_node32[i];
                                }
                                else
                                {
                                    newNode->separators[g] = UINT_MAX;
                                    newNode->childsPtrs[g] = nullptr;
                                    g++;
                                }
                            }

                            *prevLvlPtrs[j] = (void *) newNode;

                            fill(producedSeparators.begin(), producedSeparators.end(), 0);
                            fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                            fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                            fill(bounds_node16.begin(), bounds_node16.end(), 0);
                            fill(bounds_node32.begin(), bounds_node32.end(), 0);
                            
                            break;
                        }
                        else
                        {
                            expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
                            extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                            producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                            if (producedSeparators[2] > producedSeparators[0])
                            {
                                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                                newNode->dim = dim;
                                newNode->type = N64;
                                newNode->slotuse = producedSeparators[2];

                                for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                                {
                                    if (separators_node64[i])
                                    {
                                        newNode->separators[k] = separators_node64[i];
                                        currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                        nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                        numNextLvlNodes++;
                                        k++;
                                        startBound = bounds_node64[i];
                                    }
                                    else
                                    {
                                        newNode->separators[g] = ULONG_MAX;
                                        newNode->childsPtrs[g] = nullptr;
                                        g++;
                                    }
                                }

                                *prevLvlPtrs[j] = (void *) newNode;

                                fill(producedSeparators.begin(), producedSeparators.end(), 0);
                                fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                                fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                                fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                                fill(bounds_node16.begin(), bounds_node16.end(), 0);
                                fill(bounds_node32.begin(), bounds_node32.end(), 0);
                                fill(bounds_node64.begin(), bounds_node64.end(), 0);
                                
                                break;
                            }
                        }
                    }

                    verifyNode(points, dim, producedSeparators[0], separators_node16, bounds_node16, startBound, endBound);

                    Node16 *newNode = static_cast<Node16 *>(internalNodePool->allocate());
                    newNode->dim = dim;
                    newNode->type = N16;
                    newNode->slotuse = producedSeparators[0];

                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint16_t) (separators_node16[i] >> NODE_16_TRAILING_ZEROS);
                        currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node16[i]);
                        numNextLvlNodes++;
                        startBound = bounds_node16[i];
                    }

                    for (; i < NODE_SIZE_16; i++)
                    {
                        newNode->separators[i] = USHRT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node16.begin(), bounds_node16.end(), 0);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);
                    
                    break;
                }
                case N32:
                {
                    uint32_t expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                    extractSeparatorsNode32(points, dim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    if (producedSeparators[1] >= NODE_SIZE_64)
                    {
                        Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                        newNode->dim = dim;
                        newNode->type = N32;
                        newNode->slotuse = producedSeparators[1];

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            if (separators_node32[i])
                            {
                                newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                numNextLvlNodes++;
                                k++;
                                startBound = bounds_node32[i];
                            }
                            else
                            {
                                newNode->separators[g] = UINT_MAX;
                                newNode->childsPtrs[g] = nullptr;
                                g++;
                            }
                        }

                        *prevLvlPtrs[j] = (void *) newNode;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                        fill(bounds_node32.begin(), bounds_node32.end(), 0);

                        break;
                    }
                    else
                    {
                        expectedSeparators = NODE_SIZE_64;
                        extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                        producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                        if (producedSeparators[2] > producedSeparators[1])
                        {
                            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                            newNode->dim = dim;
                            newNode->type = N64;
                            newNode->slotuse = producedSeparators[2];

                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                            {
                                if (separators_node64[i])
                                {
                                    newNode->separators[k] = separators_node64[i];
                                    currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                    numNextLvlNodes++;
                                    k++;
                                    startBound = bounds_node64[i];
                                }
                                else
                                {
                                    newNode->separators[g] = ULONG_MAX;
                                    newNode->childsPtrs[g] = nullptr;
                                    g++;
                                }
                            }

                            *prevLvlPtrs[j] = (void *) newNode;

                            fill(producedSeparators.begin(), producedSeparators.end(), 0);
                            fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                            fill(bounds_node32.begin(), bounds_node32.end(), 0);
                            fill(bounds_node64.begin(), bounds_node64.end(), 0);

                            break;
                        }
                    }

                    verifyNode(points, dim, producedSeparators[1], separators_node32, bounds_node32, startBound, endBound);
                    
                    Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                    newNode->dim = dim;
                    newNode->type = N32;
                    newNode->slotuse = producedSeparators[1];

                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                        currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                        numNextLvlNodes++;
                        startBound = bounds_node32[i];
                    }

                    for (; i < NODE_SIZE_32; i++)
                    {
                        newNode->separators[i] = UINT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);

                    break;
                }
                case N64:
                {
                    uint32_t expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
                    extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                    newNode->dim = dim;
                    newNode->type = N64;
                    newNode->slotuse = producedSeparators[2];

                    for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            newNode->separators[k] = separators_node64[i];
                            currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                            nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node64[i];
                        }
                        else
                        {
                            newNode->separators[g] = ULONG_MAX;
                            newNode->childsPtrs[g] = nullptr;
                            g++;
                        }
                    }

                    *prevLvlPtrs[j] = (void *) newNode;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);
                    break;
                }
            }
        }

        prevLvlPtrs = move(currLvlPtrs);
        currLvlPtrs.clear();

        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;
    }

    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->dim = dim;
        newNode->type = Leaf;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            newNode->records[i].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = p[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }

}

template<size_t Dim>
inline void collect_points_3way_nodes(void *root, types::Points<Dim> &points)
{
    uint8_t dim = 0, type;
    uint32_t metadata;
    stack<void*> stack;
    stack.push(root);
    while (!stack.empty())
    {
        void *node = stack.top();
        stack.pop();
        metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N16:
            {
                Node16 *actualNode = reinterpret_cast<Node16 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = currLeaf->slotuse;
                uint32_t offset = points.size();

                points.resize(numRecords + offset);
                for (uint32_t i = 0; i < numRecords; i++)
                {
                    types::Point<Dim> p;

                    for (uint32_t j = 0; j < Dim; j++)
                    {
                        p[j] = currLeaf->records[j][i];
                    }

                    points[i + offset] = p;
                }
                break;
            }
        }
    }

}

template <size_t Dim>
inline bool reconstruct_subtree_3way_nodes(void *root, uint32_t subtreeDepth, types::Point<Dim> point)
{
    uint8_t dim, type;
    uint32_t metadata = 0;
    types::Points<Dim> subtreePoints;

    subtreePoints.emplace_back(point);
    
    metadata = *reinterpret_cast<uint32_t *>(root);
    type = metadata & 0x3;
    dim = (metadata >> 2) & 0x3F;

    switch (type)
    {
        case NodeType::N16:
        {
            Node16 *actualRootNode = reinterpret_cast<Node16 *>(root);
            uint32_t pos = successorLinGNode16(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));

            collect_points_3way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);

            if (actualRootNode->slotuse == NODE_SIZE_16)
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t medianPos = subtreePoints.size() >> 1;
            auto lowIt = subtreePoints.begin();
            auto medianIt = subtreePoints.begin() + medianPos;
            auto highIt = subtreePoints.end();

            nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});
            
            uint64_t convertedElement = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
            uint64_t mask = ((uint64_t) USHRT_MAX << NODE_16_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;

            auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) { 
                    return p[dim] < countingThreshold;
                }
            );

            uint32_t leftRegion = partitionIt - lowIt;
            uint32_t rightRegion = highIt - partitionIt;
            
            if (leftRegion < (leafCapacity >> 1))
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t slotuse = actualRootNode->slotuse;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint16_t));
            actualRootNode->separators[pos] = (uint16_t) (newSeparator >> NODE_16_TRAILING_ZEROS);
            
            void *newSubtree = actualRootNode->childsPtrs[slotuse];
            memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

            if (newSubtree == nullptr)
            {
                // update right subtree (i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // create left subtree (i.e., &ptr[pos])
                create_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
            }
            else
            {

                // update right subtree(i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
    
                // update left subtree(i.e., ptr[pos])
                actualRootNode->childsPtrs[pos] = newSubtree;
                subtreeDims.clear();
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
    
            }

            actualRootNode->slotuse++;
            break;
        }
        case NodeType::N32:
        {
            Node32 *actualRootNode = reinterpret_cast<Node32 *>(root);
            uint32_t pos = successorLinGNode32(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));
            
            collect_points_3way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);

            if (actualRootNode->slotuse == NODE_SIZE_32)
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);

                return true;
            }

            uint32_t medianPos = subtreePoints.size() >> 1;            
            auto lowIt = subtreePoints.begin();
            auto medianIt = subtreePoints.begin() + medianPos;
            auto highIt = subtreePoints.end();

            nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});

            uint64_t convertedElement = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
            uint64_t mask = ((uint64_t) UINT_MAX << NODE_32_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;

            auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) {
                    return p[dim] < countingThreshold;
                }
            );

            uint32_t leftRegion = partitionIt - lowIt;
            uint32_t rightRegion = highIt - partitionIt;

            if (leftRegion < (leafCapacity >> 1))
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t slotuse = actualRootNode->slotuse;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint32_t));
            actualRootNode->separators[pos] = (uint32_t) (newSeparator >> NODE_32_TRAILING_ZEROS);
            
            void *newSubtree = actualRootNode->childsPtrs[slotuse];
            memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

            if (newSubtree == nullptr)
            {
                vector<uint8_t> subtreeDims;
                // update right subtree (i.e., ptr[pos+1])
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // create left subtree (i.e., &ptr[pos])
                create_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
             
            }
            else
            {
                // update right subtree(i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // update left subtree(i.e., ptr[pos])
                actualRootNode->childsPtrs[pos] = newSubtree;
                subtreeDims.clear();
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
            }

            actualRootNode->slotuse++;
            break;
        }
        case NodeType::N64:
        {
            Node64 *actualRootNode = reinterpret_cast<Node64 *>(root);
            uint32_t pos = successorLinGNode64(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));
            
            collect_points_3way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);
            
            if (actualRootNode->slotuse == NODE_SIZE_64)
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t medianPos = subtreePoints.size() >> 1;            
            auto lowIt = subtreePoints.begin();
            auto medianIt = subtreePoints.begin() + medianPos;
            auto highIt = subtreePoints.end();

            nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});

            uint64_t newSeparator = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
            double countingThreshold = subtreePoints[medianPos][dim];
            
            auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) {
                    return p[dim] < countingThreshold;
                }
            );

            uint32_t leftRegion = partitionIt - lowIt;
            uint32_t rightRegion = highIt - partitionIt;

            if (leftRegion < (leafCapacity >> 1))
            {
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t slotuse = actualRootNode->slotuse;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint64_t));
            actualRootNode->separators[pos] = newSeparator;

            void *newSubtree = actualRootNode->childsPtrs[slotuse];
            memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

            if (newSubtree == nullptr)
            {
                vector<uint8_t> subtreeDims;
                // update right subtree (i.e., ptr[pos+1])
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // create left subtree (i.e., &ptr[pos])
                create_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
             
            }
            else
            {
            
                // update right subtree(i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
            
                // update left subtree(i.e., ptr[pos])
                actualRootNode->childsPtrs[pos] = newSubtree;
                subtreeDims.clear();
                update_subtree_3way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
            }

            actualRootNode->slotuse++;  

            break;
        }
    }

    return true;
}

template <size_t Dim>
inline void make_split_3way_nodes(void *parentNode, LeafNode<Dim> *currLeaf, uint32_t parentMetadata, uint32_t pos)
{
    uint8_t type = parentMetadata & 0x3;
    uint8_t dim = (parentMetadata >> 2) & 0x3F;
    
    vector<uint32_t> indices(currLeaf->slotuse);
    iota(indices.begin(), indices.end(), 0);

    uint32_t medianPos = indices.size() >> 1;

    nth_element(indices.begin(), indices.begin() + medianPos, indices.end(),
                [&](uint32_t idx1, uint32_t idx2) {
                    return currLeaf->records[dim][idx1] < currLeaf->records[dim][idx2];
                }
    );

    uint64_t convertedElement = (uint64_t) (currLeaf->records[dim][indices[medianPos]] * CONVERSION_FACTOR);

    switch (type)
    {
        case NodeType::N16:
        {
            Node16 *actualParentNode = reinterpret_cast<Node16 *>(parentNode);
            uint64_t mask = ((uint64_t) USHRT_MAX << NODE_16_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;
            
            auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                        return currLeaf->records[dim][idx] < countingThreshold;
                    }
            );

            size_t leftRegion = partitionIt - indices.begin();
            size_t rightRegion = indices.size() - leftRegion;
            
            if (leftRegion < (leafCapacity >> 1))
            {
                if (!currLeaf->is_outlier)
                {
                    currLeaf->is_outlier = 1;
                }
                else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
                    treeStats.maxLeafCapacity = currLeaf->slotuse;

                types::Point<Dim> minCorner, maxCorner;
                for (uint32_t j = 0; j < Dim; j++)
                {

                    auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
                    minCorner[j] = *minIt;
                    maxCorner[j] = *maxIt;
                }

                currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

                return;
            }

            void *newNode = nullptr;
            LeafNode<Dim> *newLeaf = nullptr;

            uint32_t slotuse = actualParentNode->slotuse;
            actualParentNode->slotuse++;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint16_t));
            actualParentNode->separators[pos] = (uint16_t) (newSeparator >> NODE_16_TRAILING_ZEROS);
            
            newNode = actualParentNode->childsPtrs[slotuse];
            memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

            // There isn't an empty linked node from previous re-partitioning
            if (newNode == nullptr)
            {
                actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
                newLeaf = new (newNode) LeafNode<Dim>();
            }
            else
            {
                actualParentNode->childsPtrs[pos+1] = newNode;
                newLeaf = new (newNode) LeafNode<Dim>();
                newLeaf->slotuse = 0;
                newLeaf->is_outlier = 0;

                for (uint32_t d = 0; d < Dim; d++)
                {
                    newLeaf->records[d].clear();
                }
            }

            // Fill the vectors of each leaf node with the appropriate coordinates
            sort(indices.begin(), indices.begin() + leftRegion);

            for (uint32_t j = 0; j < Dim; j++)
            {
                newLeaf->records[j].resize(rightRegion);
                for (uint32_t i = 0; i < rightRegion; i++)
                {
                    uint32_t idx = indices[leftRegion + i];
                    newLeaf->records[j][i] = currLeaf->records[j][idx];
                }

                for (uint32_t i = 0; i < leftRegion; i++)
                {
                    uint32_t idx = indices[i];
                    currLeaf->records[j][i] = currLeaf->records[j][idx];
                }
            }

            currLeaf->slotuse = leftRegion;
            newLeaf->slotuse = rightRegion;

            if (leftRegion < treeStats.splitThreshold)
            {
                currLeaf->is_outlier = 0;
                leftRegion = treeStats.splitThreshold;
            }
            else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
            {
                currLeaf->is_outlier = 0;
            }
            else
            {
                currLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
            }
            
            if (rightRegion < treeStats.splitThreshold)
            {
                newLeaf->is_outlier = 0;
                rightRegion = treeStats.splitThreshold;
            }
            else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
            {
                newLeaf->is_outlier = 0;
            }
            else
            {
                newLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
            }

            types::Point<Dim> oldMinCorner, oldMaxCorner;
            types::Point<Dim> newMinCorner, newMaxCorner;

            // Update the BBs and vector sizes
            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j].resize(leftRegion);
                auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

                oldMinCorner[j] = *oldMinIt;
                oldMaxCorner[j] = *oldMaxIt;

                newLeaf->records[j].resize(rightRegion);
                auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

                newMinCorner[j] = *newMinIt;
                newMaxCorner[j] = *newMaxIt;
            }

            currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

            newLeaf->dim = dim;
            newLeaf->type = Leaf;
            newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

            break;
        }
        case NodeType::N32:
        {
            Node32 *actualParentNode = reinterpret_cast<Node32 *>(parentNode);
            uint64_t mask = ((uint64_t) UINT_MAX << NODE_32_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;
            
            auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                        return currLeaf->records[dim][idx] < countingThreshold;
                    }
            );

            size_t leftRegion = partitionIt - indices.begin();
            size_t rightRegion = indices.size() - leftRegion;

            if (leftRegion < (leafCapacity >> 1))
            {
                if (!currLeaf->is_outlier)
                {
                    currLeaf->is_outlier = 1;
                }
                else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
                    treeStats.maxLeafCapacity = currLeaf->slotuse;

                types::Point<Dim> minCorner, maxCorner;
                for (uint32_t j = 0; j < Dim; j++)
                {

                    auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
                    minCorner[j] = *minIt;
                    maxCorner[j] = *maxIt;
                }

                currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

                return;
            }

            void *newNode = nullptr;
            LeafNode<Dim> *newLeaf = nullptr;

            uint32_t slotuse = actualParentNode->slotuse;
            actualParentNode->slotuse++;
            uint32_t shiftedEntries = slotuse - pos;


            memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint32_t));
            actualParentNode->separators[pos] = (uint32_t) (newSeparator >> NODE_32_TRAILING_ZEROS);
            
            newNode = actualParentNode->childsPtrs[slotuse];
            memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

            // There isn't an empty linked node from previous re-partitioning
            if (newNode == nullptr)
            {
                actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
                newLeaf = new (newNode) LeafNode<Dim>();
            }
            else
            {
                actualParentNode->childsPtrs[pos+1] = newNode;
                newLeaf = new (newNode) LeafNode<Dim>();
                newLeaf->slotuse = 0;
                newLeaf->is_outlier = 0;

                for (uint32_t d = 0; d < Dim; d++)
                {
                    newLeaf->records[d].clear();
                }
            }

            // Fill the vectors of each leaf node with the appropriate coordinates
            sort(indices.begin(), indices.begin() + leftRegion);

            for (uint32_t j = 0; j < Dim; j++)
            {
                newLeaf->records[j].resize(rightRegion);
                for (uint32_t i = 0; i < rightRegion; i++)
                {
                    uint32_t idx = indices[leftRegion + i];
                    newLeaf->records[j][i] = currLeaf->records[j][idx];
                }

                for (uint32_t i = 0; i < leftRegion; i++)
                {
                    uint32_t idx = indices[i];
                    currLeaf->records[j][i] = currLeaf->records[j][idx];
                }
            }

            currLeaf->slotuse = leftRegion;
            newLeaf->slotuse = rightRegion;

            if (leftRegion < treeStats.splitThreshold)
            {
                currLeaf->is_outlier = 0;
                leftRegion = treeStats.splitThreshold;
            }
            else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
            {
                currLeaf->is_outlier = 0;
            }
            else
            {
                currLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
            }
            
            if (rightRegion < treeStats.splitThreshold)
            {
                newLeaf->is_outlier = 0;
                rightRegion = treeStats.splitThreshold;
            }
            else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
            {
                newLeaf->is_outlier = 0;
            }
            else
            {
                newLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
            }

            types::Point<Dim> oldMinCorner, oldMaxCorner;
            types::Point<Dim> newMinCorner, newMaxCorner;

            // Update the BBs and vector sizes
            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j].resize(leftRegion);
                auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

                oldMinCorner[j] = *oldMinIt;
                oldMaxCorner[j] = *oldMaxIt;

                newLeaf->records[j].resize(rightRegion);
                auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

                newMinCorner[j] = *newMinIt;
                newMaxCorner[j] = *newMaxIt;
            }

            currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

            newLeaf->dim = dim;
            newLeaf->type = Leaf;
            newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

            break;
        }
        case NodeType::N64:
        {
            Node64 *actualParentNode = reinterpret_cast<Node64 *>(parentNode);
            uint64_t newSeparator = convertedElement;
            double countingThreshold = currLeaf->records[dim][indices[medianPos]];

            auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                        return currLeaf->records[dim][idx] < countingThreshold;
                    }
            );

            size_t leftRegion = partitionIt - indices.begin();
            size_t rightRegion = indices.size() - leftRegion;

            if (leftRegion < (leafCapacity >> 1))
            {
                if (!currLeaf->is_outlier)
                {
                    currLeaf->is_outlier = 1;
                }
                else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
                    treeStats.maxLeafCapacity = currLeaf->slotuse;

                types::Point<Dim> minCorner, maxCorner;
                for (uint32_t j = 0; j < Dim; j++)
                {

                    auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
                    minCorner[j] = *minIt;
                    maxCorner[j] = *maxIt;
                }

                currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

                return;
            }            

            void *newNode = nullptr;
            LeafNode<Dim> *newLeaf = nullptr;

            uint32_t slotuse = actualParentNode->slotuse;
            actualParentNode->slotuse++;
            uint32_t shiftedEntries = slotuse - pos;


            memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint64_t));
            actualParentNode->separators[pos] = newSeparator;
            
            newNode = actualParentNode->childsPtrs[slotuse];
            memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

            // There isn't an empty linked node from previous re-partitioning
            if (newNode == nullptr)
            {
                actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
                newLeaf = new (newNode) LeafNode<Dim>();
            }
            else
            {
                actualParentNode->childsPtrs[pos+1] = newNode;
                newLeaf = new (newNode) LeafNode<Dim>();
                newLeaf->slotuse = 0;
                newLeaf->is_outlier = 0;

                for (uint32_t d = 0; d < Dim; d++)
                {
                    newLeaf->records[d].clear();
                }
            }

            // Fill the vectors of each leaf node with the appropriate coordinates
            sort(indices.begin(), indices.begin() + leftRegion);

            for (uint32_t j = 0; j < Dim; j++)
            {
                newLeaf->records[j].resize(rightRegion);
                for (uint32_t i = 0; i < rightRegion; i++)
                {
                    uint32_t idx = indices[leftRegion + i];
                    newLeaf->records[j][i] = currLeaf->records[j][idx];
                }

                for (uint32_t i = 0; i < leftRegion; i++)
                {
                    uint32_t idx = indices[i];
                    currLeaf->records[j][i] = currLeaf->records[j][idx];
                }
            }
   
            currLeaf->slotuse = leftRegion;
            newLeaf->slotuse = rightRegion;
         
            if (leftRegion < treeStats.splitThreshold)
            {
                currLeaf->is_outlier = 0;
                leftRegion = treeStats.splitThreshold;
            }
            else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
            {
                currLeaf->is_outlier = 0;
            }
            else
            {
                currLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
            }
            
            if (rightRegion < treeStats.splitThreshold)
            {
                newLeaf->is_outlier = 0;
                rightRegion = treeStats.splitThreshold;
            }
            else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
            {
                newLeaf->is_outlier = 0;
            }
            else
            {
                newLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
            }


            types::Point<Dim> oldMinCorner, oldMaxCorner;
            types::Point<Dim> newMinCorner, newMaxCorner;

            // Update the BBs and vector sizes
            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j].resize(leftRegion);
                auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

                oldMinCorner[j] = *oldMinIt;
                oldMaxCorner[j] = *oldMaxIt;

                newLeaf->records[j].resize(rightRegion);
                auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

                newMinCorner[j] = *newMinIt;
                newMaxCorner[j] = *newMaxIt;
            }

            currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

            newLeaf->dim = dim;
            newLeaf->type = Leaf;
            newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

            break;
        }
    }
}


template<size_t Dim>
inline void update_subtree_2way_nodes(void *root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t>& dimPerLvl)
{  
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector<uint32_t> bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void *> currLvlNodes, nextLvlNodes;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    currLvlNodes.emplace_back(root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level; l < treeStats.treeDepth; l++)
    {
        numNextLvlNodes = 0;

        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else 
        {
            exponent = treeStats.treeDepth - l;
        }

        for (j = 0; j < numNodesPerLevel; j++)
        {
            void *node = currLvlNodes[j];
            metadata = *reinterpret_cast<uint32_t *>(node);
            nodeType = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;

            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});

            uint32_t subspaceLeaves = round(numOfElements/(double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));

            switch (nodeType)
            {
                case NodeType::N32:
                {
                    Node32 *actualNode = reinterpret_cast<Node32 *>(node);

                    if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
                    {
                        actualNode->slotuse = 1;
                        actualNode->separators[0] = UINT_MAX;
                        nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                        nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                        numNextLvlNodes++;

                        for (i = actualNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            actualNode->separators[i] = UINT_MAX;
                        }

                        break;
                    }
                    
                    uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
                    extractSeparatorsNode32(points, dim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    uint32_t producedSeparators = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    actualNode->slotuse = producedSeparators;
                
                    for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_32; i++)
                    {
                        if (separators_node32[i])
                        {
                            actualNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                            nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                            nextLvlBounds.emplace_back(make_pair(startBound, bounds_node32[i]));
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node32[i];
                        }
                        else
                        {
                            actualNode->separators[g] = UINT_MAX;
                            g++;
                        }
                    }

                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);

                    break;
                }
                case NodeType::N64:
                {
                    Node64 *actualNode = reinterpret_cast<Node64 *>(node);

                    if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
                    {
                        actualNode->slotuse = 1;
                        actualNode->separators[0] = ULONG_MAX;
                        nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                        nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                        numNextLvlNodes++;

                        for (i = actualNode->slotuse; i < NODE_SIZE_64; i++)
                        {
                            actualNode->separators[i] = ULONG_MAX;
                        }

                        break;
                    }

                    uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
                    extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    uint32_t producedSeparators = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    actualNode->slotuse = producedSeparators;
                
                    for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            actualNode->separators[k] = separators_node64[i];
                            nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                            nextLvlBounds.emplace_back(make_pair(startBound, bounds_node64[i]));
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node64[i];
                        }
                        else
                        {
                            actualNode->separators[g] = ULONG_MAX;
                            g++;
                        }
                    }

                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);
                    
                    break;
                }
            }
        }

        currLvlNodes = move(nextLvlNodes);
        nextLvlNodes.clear();
        
        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;

        dimPerLvl.emplace_back(dim);

    }


    for (j = 0; j < numNodesPerLevel; j++)
    {

        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(currLvlNodes[j]);
        dim = currLeaf->dim;


        if (numOfElements <= treeStats.splitThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            currLeaf->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            currLeaf->records[i].resize(leafNodeSize);
        }

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];

            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j][k] = p[j];
            }
        }

        currLeaf->slotuse = numOfElements;
        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);
    }
}

template<size_t Dim>
inline void create_subtree_2way_nodes(void *&root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t> dimPerLvl)
{
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    prevLvlPtrs.emplace_back(&root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level, d = 0; l < treeStats.treeDepth; l++, d++)
    {
        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else
        {
            exponent = treeStats.treeDepth - l;
        }

        numNextLvlNodes = 0;
        currLvlPtrs.resize(NODE_SIZE_32 * numNodesPerLevel);
        nextLvlBounds.resize(NODE_SIZE_32 * numNodesPerLevel);

        dim = dimPerLvl[d];

        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});
            uint32_t subspaceLeaves = round(numOfElements / (double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));

            if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
            {
                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                newNode->dim = dim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, endBound);
                numNextLvlNodes++;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }

                *prevLvlPtrs[j] = (void *) newNode;

                continue;
            }

            if (localSplits > NODE_SIZE_64)
                nodeType = N32;
            else
                nodeType = N64;


            switch (nodeType)
            {
                case N32:
                {
                    uint32_t expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                    extractSeparatorsNode32(points, dim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    if (producedSeparators[1] >= NODE_SIZE_64)
                    {
                        Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                        newNode->dim = dim;
                        newNode->type = N32;
                        newNode->slotuse = producedSeparators[1];

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            if (separators_node32[i])
                            {
                                newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                numNextLvlNodes++;
                                k++;
                                startBound = bounds_node32[i];
                            }
                            else
                            {
                                newNode->separators[g] = UINT_MAX;
                                newNode->childsPtrs[g] = nullptr;
                                g++;
                            }
                        }

                        *prevLvlPtrs[j] = (void *) newNode;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                        fill(bounds_node32.begin(), bounds_node32.end(), 0);

                        break;
                    }
                    else
                    {
                        expectedSeparators = NODE_SIZE_64;
                        extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                        producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                        if (producedSeparators[2] > producedSeparators[1])
                        {
                            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                            newNode->dim = dim;
                            newNode->type = N64;
                            newNode->slotuse = producedSeparators[2];

                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                            {
                                if (separators_node64[i])
                                {
                                    newNode->separators[k] = separators_node64[i];
                                    currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                    numNextLvlNodes++;
                                    k++;
                                    startBound = bounds_node64[i];
                                }
                                else
                                {
                                    newNode->separators[g] = ULONG_MAX;
                                    newNode->childsPtrs[g] = nullptr;
                                    g++;
                                }
                            }

                            *prevLvlPtrs[j] = (void *) newNode;

                            fill(producedSeparators.begin(), producedSeparators.end(), 0);
                            fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                            fill(bounds_node32.begin(), bounds_node32.end(), 0);
                            fill(bounds_node64.begin(), bounds_node64.end(), 0);

                            break;
                        }
                    }

                    verifyNode(points, dim, producedSeparators[1], separators_node32, bounds_node32, startBound, endBound);
                    
                    Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                    newNode->dim = dim;
                    newNode->type = N32;
                    newNode->slotuse = producedSeparators[1];

                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                        currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                        numNextLvlNodes++;
                        startBound = bounds_node32[i];
                    }

                    for (; i < NODE_SIZE_32; i++)
                    {
                        newNode->separators[i] = UINT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);

                    break;
                }
                case N64:
                {
                    uint32_t expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
                    extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                    newNode->dim = dim;
                    newNode->type = N64;
                    newNode->slotuse = producedSeparators[2];

                    for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            newNode->separators[k] = separators_node64[i];
                            currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                            nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                            numNextLvlNodes++;
                            k++;
                            startBound = bounds_node64[i];
                        }
                        else
                        {
                            newNode->separators[g] = ULONG_MAX;
                            newNode->childsPtrs[g] = nullptr;
                            g++;
                        }
                    }

                    *prevLvlPtrs[j] = (void *) newNode;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);
                    break;
                }
            }
        }

        prevLvlPtrs = move(currLvlPtrs);
        currLvlPtrs.clear();

        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;
    }

    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->dim = dim;
        newNode->type = Leaf;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            newNode->records[i].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = p[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }

}

template<size_t Dim>
inline void collect_points_2way_nodes(void *root, types::Points<Dim> &points)
{
    uint8_t dim = 0, type;
    uint32_t metadata;
    stack<void*> stack;
    stack.push(root);
    while (!stack.empty())
    {
        void *node = stack.top();
        stack.pop();
        metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = currLeaf->slotuse;
                uint32_t offset = points.size();

                points.resize(numRecords + offset);
                for (uint32_t i = 0; i < numRecords; i++)
                {
                    types::Point<Dim> p;

                    for (uint32_t j = 0; j < Dim; j++)
                    {
                        p[j] = currLeaf->records[j][i];
                    }

                    points[i + offset] = p;
                }
                break;
            }
        }
    }

}

template <size_t Dim>
inline bool reconstruct_subtree_2way_nodes(void *root, uint32_t subtreeDepth, types::Point<Dim> point)
{
    uint8_t dim, type;
    uint32_t metadata = 0;
    types::Points<Dim> subtreePoints;

    subtreePoints.emplace_back(point);
    
    metadata = *reinterpret_cast<uint32_t *>(root);
    type = metadata & 0x3;
    dim = (metadata >> 2) & 0x3F;

    switch (type)
    {
        case NodeType::N32:
        {
            Node32 *actualRootNode = reinterpret_cast<Node32 *>(root);
            uint32_t pos = successorLinGNode32(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));
            
            collect_points_2way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);

            if (actualRootNode->slotuse == NODE_SIZE_32)
            {
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);

                return true;
            }

            uint32_t medianPos = subtreePoints.size() >> 1;            
            auto lowIt = subtreePoints.begin();
            auto medianIt = subtreePoints.begin() + medianPos;
            auto highIt = subtreePoints.end();

            nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});

            uint64_t convertedElement = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
            uint64_t mask = ((uint64_t) UINT_MAX << NODE_32_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;

            auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) {
                    return p[dim] < countingThreshold;
                }
            );

            uint32_t leftRegion = partitionIt - lowIt;
            uint32_t rightRegion = highIt - partitionIt;

            if (leftRegion < (leafCapacity >> 1))
            {
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t slotuse = actualRootNode->slotuse;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint32_t));
            actualRootNode->separators[pos] = (uint32_t) (newSeparator >> NODE_32_TRAILING_ZEROS);
            
            void *newSubtree = actualRootNode->childsPtrs[slotuse];
            memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

            if (newSubtree == nullptr)
            {
                vector<uint8_t> subtreeDims;
                // update right subtree (i.e., ptr[pos+1])
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // create left subtree (i.e., &ptr[pos])
                create_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
             
            }
            else
            {
                // update right subtree(i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // update left subtree(i.e., ptr[pos])
                actualRootNode->childsPtrs[pos] = newSubtree;
                subtreeDims.clear();
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
            }

            actualRootNode->slotuse++;
            break;
        }
        case NodeType::N64:
        {
            Node64 *actualRootNode = reinterpret_cast<Node64 *>(root);
            uint32_t pos = successorLinGNode64(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));
            
            collect_points_2way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);
            
            if (actualRootNode->slotuse == NODE_SIZE_64)
            {
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t medianPos = subtreePoints.size() >> 1;            
            auto lowIt = subtreePoints.begin();
            auto medianIt = subtreePoints.begin() + medianPos;
            auto highIt = subtreePoints.end();

            nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});

            uint64_t newSeparator = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
            double countingThreshold = subtreePoints[medianPos][dim];
            
            auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) {
                    return p[dim] < countingThreshold;
                }
            );

            uint32_t leftRegion = partitionIt - lowIt;
            uint32_t rightRegion = highIt - partitionIt;

            if (leftRegion < (leafCapacity >> 1))
            {
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
                return true;
            }

            uint32_t slotuse = actualRootNode->slotuse;
            uint32_t shiftedEntries = slotuse - pos;

            memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint64_t));
            actualRootNode->separators[pos] = newSeparator;

            void *newSubtree = actualRootNode->childsPtrs[slotuse];
            memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

            if (newSubtree == nullptr)
            {
                vector<uint8_t> subtreeDims;
                // update right subtree (i.e., ptr[pos+1])
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
             
                // create left subtree (i.e., &ptr[pos])
                create_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
             
            }
            else
            {
            
                // update right subtree(i.e., ptr[pos+1])
                vector<uint8_t> subtreeDims;
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
            
                // update left subtree(i.e., ptr[pos])
                actualRootNode->childsPtrs[pos] = newSubtree;
                subtreeDims.clear();
                update_subtree_2way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
            }

            actualRootNode->slotuse++;  

            break;
        }
    }

    return true;
}

template <size_t Dim>
inline void make_split_2way_nodes(void *parentNode, LeafNode<Dim> *currLeaf, uint32_t parentMetadata, uint32_t pos)
{
    uint8_t type = parentMetadata & 0x3;
    uint8_t dim = (parentMetadata >> 2) & 0x3F;
    
    vector<uint32_t> indices(currLeaf->slotuse);
    iota(indices.begin(), indices.end(), 0);

    uint32_t medianPos = indices.size() >> 1;

    nth_element(indices.begin(), indices.begin() + medianPos, indices.end(),
                [&](uint32_t idx1, uint32_t idx2) {
                    return currLeaf->records[dim][idx1] < currLeaf->records[dim][idx2];
                }
    );

    uint64_t convertedElement = (uint64_t) (currLeaf->records[dim][indices[medianPos]] * CONVERSION_FACTOR);

    switch (type)
    {
        case NodeType::N32:
        {
            Node32 *actualParentNode = reinterpret_cast<Node32 *>(parentNode);
            uint64_t mask = ((uint64_t) UINT_MAX << NODE_32_TRAILING_ZEROS);
            uint64_t newSeparator = convertedElement & mask;
            double countingThreshold = newSeparator / (double) CONVERSION_FACTOR;
            
            auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                        return currLeaf->records[dim][idx] < countingThreshold;
                    }
            );

            size_t leftRegion = partitionIt - indices.begin();
            size_t rightRegion = indices.size() - leftRegion;

            if (leftRegion < (leafCapacity >> 1))
            {
                if (!currLeaf->is_outlier)
                {
                    currLeaf->is_outlier = 1;
                }
                else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
                    treeStats.maxLeafCapacity = currLeaf->slotuse;

                types::Point<Dim> minCorner, maxCorner;
                for (uint32_t j = 0; j < Dim; j++)
                {

                    auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
                    minCorner[j] = *minIt;
                    maxCorner[j] = *maxIt;
                }

                currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

                return;
            }

            void *newNode = nullptr;
            LeafNode<Dim> *newLeaf = nullptr;

            uint32_t slotuse = actualParentNode->slotuse;
            actualParentNode->slotuse++;
            uint32_t shiftedEntries = slotuse - pos;


            memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint32_t));
            actualParentNode->separators[pos] = (uint32_t) (newSeparator >> NODE_32_TRAILING_ZEROS);
            
            newNode = actualParentNode->childsPtrs[slotuse];
            memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

            // There isn't an empty linked node from previous re-partitioning
            if (newNode == nullptr)
            {
                actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
                newLeaf = new (newNode) LeafNode<Dim>();
            }
            else
            {
                actualParentNode->childsPtrs[pos+1] = newNode;
                newLeaf = new (newNode) LeafNode<Dim>();
                newLeaf->slotuse = 0;
                newLeaf->is_outlier = 0;

                for (uint32_t d = 0; d < Dim; d++)
                {
                    newLeaf->records[d].clear();
                }
            }

            // Fill the vectors of each leaf node with the appropriate coordinates
            sort(indices.begin(), indices.begin() + leftRegion);

            for (uint32_t j = 0; j < Dim; j++)
            {
                newLeaf->records[j].resize(rightRegion);
                for (uint32_t i = 0; i < rightRegion; i++)
                {
                    uint32_t idx = indices[leftRegion + i];
                    newLeaf->records[j][i] = currLeaf->records[j][idx];
                }

                for (uint32_t i = 0; i < leftRegion; i++)
                {
                    uint32_t idx = indices[i];
                    currLeaf->records[j][i] = currLeaf->records[j][idx];
                }
            }

            currLeaf->slotuse = leftRegion;
            newLeaf->slotuse = rightRegion;

            if (leftRegion < treeStats.splitThreshold)
            {
                currLeaf->is_outlier = 0;
                leftRegion = treeStats.splitThreshold;
            }
            else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
            {
                currLeaf->is_outlier = 0;
            }
            else
            {
                currLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
            }
            
            if (rightRegion < treeStats.splitThreshold)
            {
                newLeaf->is_outlier = 0;
                rightRegion = treeStats.splitThreshold;
            }
            else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
            {
                newLeaf->is_outlier = 0;
            }
            else
            {
                newLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
            }

            types::Point<Dim> oldMinCorner, oldMaxCorner;
            types::Point<Dim> newMinCorner, newMaxCorner;

            // Update the BBs and vector sizes
            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j].resize(leftRegion);
                auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

                oldMinCorner[j] = *oldMinIt;
                oldMaxCorner[j] = *oldMaxIt;

                newLeaf->records[j].resize(rightRegion);
                auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

                newMinCorner[j] = *newMinIt;
                newMaxCorner[j] = *newMaxIt;
            }

            currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

            newLeaf->dim = dim;
            newLeaf->type = Leaf;
            newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

            break;
        }
        case NodeType::N64:
        {
            Node64 *actualParentNode = reinterpret_cast<Node64 *>(parentNode);
            uint64_t newSeparator = convertedElement;
            double countingThreshold = currLeaf->records[dim][indices[medianPos]];

            auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                        return currLeaf->records[dim][idx] < countingThreshold;
                    }
            );

            size_t leftRegion = partitionIt - indices.begin();
            size_t rightRegion = indices.size() - leftRegion;

            if (leftRegion < (leafCapacity >> 1))
            {
                if (!currLeaf->is_outlier)
                {
                    currLeaf->is_outlier = 1;
                }
                else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
                    treeStats.maxLeafCapacity = currLeaf->slotuse;

                types::Point<Dim> minCorner, maxCorner;
                for (uint32_t j = 0; j < Dim; j++)
                {

                    auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
                    minCorner[j] = *minIt;
                    maxCorner[j] = *maxIt;
                }

                currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

                return;
            }            

            void *newNode = nullptr;
            LeafNode<Dim> *newLeaf = nullptr;

            uint32_t slotuse = actualParentNode->slotuse;
            actualParentNode->slotuse++;
            uint32_t shiftedEntries = slotuse - pos;


            memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint64_t));
            actualParentNode->separators[pos] = newSeparator;
            
            newNode = actualParentNode->childsPtrs[slotuse];
            memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

            // There isn't an empty linked node from previous re-partitioning
            if (newNode == nullptr)
            {
                actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
                newLeaf = new (newNode) LeafNode<Dim>();
            }
            else
            {
                actualParentNode->childsPtrs[pos+1] = newNode;
                newLeaf = new (newNode) LeafNode<Dim>();
                newLeaf->slotuse = 0;
                newLeaf->is_outlier = 0;

                for (uint32_t d = 0; d < Dim; d++)
                {
                    newLeaf->records[d].clear();
                }
            }

            // Fill the vectors of each leaf node with the appropriate coordinates
            sort(indices.begin(), indices.begin() + leftRegion);

            for (uint32_t j = 0; j < Dim; j++)
            {
                newLeaf->records[j].resize(rightRegion);
                for (uint32_t i = 0; i < rightRegion; i++)
                {
                    uint32_t idx = indices[leftRegion + i];
                    newLeaf->records[j][i] = currLeaf->records[j][idx];
                }

                for (uint32_t i = 0; i < leftRegion; i++)
                {
                    uint32_t idx = indices[i];
                    currLeaf->records[j][i] = currLeaf->records[j][idx];
                }
            }
   
            currLeaf->slotuse = leftRegion;
            newLeaf->slotuse = rightRegion;
         
            if (leftRegion < treeStats.splitThreshold)
            {
                currLeaf->is_outlier = 0;
                leftRegion = treeStats.splitThreshold;
            }
            else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
            {
                currLeaf->is_outlier = 0;
            }
            else
            {
                currLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
            }
            
            if (rightRegion < treeStats.splitThreshold)
            {
                newLeaf->is_outlier = 0;
                rightRegion = treeStats.splitThreshold;
            }
            else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
            {
                newLeaf->is_outlier = 0;
            }
            else
            {
                newLeaf->is_outlier = 1;
                treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
            }


            types::Point<Dim> oldMinCorner, oldMaxCorner;
            types::Point<Dim> newMinCorner, newMaxCorner;

            // Update the BBs and vector sizes
            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j].resize(leftRegion);
                auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

                oldMinCorner[j] = *oldMinIt;
                oldMaxCorner[j] = *oldMaxIt;

                newLeaf->records[j].resize(rightRegion);
                auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

                newMinCorner[j] = *newMinIt;
                newMaxCorner[j] = *newMaxIt;
            }

            currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

            newLeaf->dim = dim;
            newLeaf->type = Leaf;
            newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

            break;
        }
    }
}


template<size_t Dim>
inline void update_subtree_1way_nodes(void *root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t>& dimPerLvl)
{  
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector<uint32_t> bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void *> currLvlNodes, nextLvlNodes;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    currLvlNodes.emplace_back(root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level; l < treeStats.treeDepth; l++)
    {
        numNextLvlNodes = 0;

        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else 
        {
            exponent = treeStats.treeDepth - l;
        }

        for (j = 0; j < numNodesPerLevel; j++)
        {
            void *node = currLvlNodes[j];
            metadata = *reinterpret_cast<uint32_t *>(node);
            nodeType = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;

            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});

            uint32_t subspaceLeaves = round(numOfElements/(double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));


             
            Node64 *actualNode = reinterpret_cast<Node64 *>(node);

            if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
            {
                actualNode->slotuse = 1;
                actualNode->separators[0] = ULONG_MAX;
                nextLvlBounds.emplace_back(make_pair(startBound, endBound));
                nextLvlNodes.emplace_back(actualNode->childsPtrs[0]);
                numNextLvlNodes++;

                for (i = actualNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    actualNode->separators[i] = ULONG_MAX;
                }

                break;
            }

            uint32_t expectedSeparators = (localSplits > actualNode->slotuse) ? actualNode->slotuse : localSplits;
            extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
            uint32_t producedSeparators = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

            actualNode->slotuse = producedSeparators;
        
            for (i = 0, k = 0, g = producedSeparators; i < NODE_SIZE_64; i++)
            {
                if (separators_node64[i])
                {
                    actualNode->separators[k] = separators_node64[i];
                    nextLvlNodes.emplace_back(actualNode->childsPtrs[k]);
                    nextLvlBounds.emplace_back(make_pair(startBound, bounds_node64[i]));
                    numNextLvlNodes++;
                    k++;
                    startBound = bounds_node64[i];
                }
                else
                {
                    actualNode->separators[g] = ULONG_MAX;
                    g++;
                }
            }

            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
            fill(bounds_node64.begin(), bounds_node64.end(), 0);
        }

        currLvlNodes = move(nextLvlNodes);
        nextLvlNodes.clear();
        
        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;

        dimPerLvl.emplace_back(dim);

    }

    for (j = 0; j < numNodesPerLevel; j++)
    {

        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(currLvlNodes[j]);
        dim = currLeaf->dim;


        if (numOfElements <= treeStats.splitThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            currLeaf->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            currLeaf->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            currLeaf->records[i].resize(leafNodeSize);
        }

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];

            for (uint32_t j = 0; j < Dim; j++)
            {
                currLeaf->records[j][k] = p[j];
            }
        }

        currLeaf->slotuse = numOfElements;
        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);
    }
}

template<size_t Dim>
inline void create_subtree_1way_nodes(void *&root, uint32_t subtreeDepth, uint32_t low, uint32_t high, types::Points<Dim> &points, vector<uint8_t> dimPerLvl)
{
    uint8_t nodeType, dim;
    uint32_t numOfElements = 0, metadata, exponent;
    uint32_t i = 0, j = 0, k = 0, g = 0;
    uint32_t numNodesPerLevel = 1, numNextLvlNodes = 0;
    uint32_t level = subtreeDepth, startBound = low, endBound = high;
    
    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node64(NODE_SIZE_64, ULONG_MAX);
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;

    prevLvlPtrs.emplace_back(&root);
    currLvlBounds.emplace_back(make_pair(startBound, endBound));

    for (uint32_t l = level, d = 0; l < treeStats.treeDepth; l++, d++)
    {
        if (l < Dim)
        {
            exponent = Dim - l;
        }
        else
        {
            exponent = treeStats.treeDepth - l;
        }

        numNextLvlNodes = 0;
        currLvlPtrs.resize(NODE_SIZE_64 * numNodesPerLevel);
        nextLvlBounds.resize(NODE_SIZE_64 * numNodesPerLevel);

        dim = dimPerLvl[d];

        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [dim](auto &a, auto&b){return a[dim] < b[dim];});
            uint32_t subspaceLeaves = round(numOfElements / (double) leafCapacity);
            uint32_t localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));

            if (localSplits == 1 || numOfElements <= treeStats.splitThreshold || (*minIt)[dim] == (*maxIt)[dim])
            {
                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                newNode->dim = dim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[numNextLvlNodes] = make_pair(startBound, endBound);
                numNextLvlNodes++;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }

                *prevLvlPtrs[j] = (void *) newNode;

                continue;
            }


            uint32_t expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
            extractSeparatorsNode64(points, dim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
            producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
            newNode->dim = dim;
            newNode->type = N64;
            newNode->slotuse = producedSeparators[2];

            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
            {
                if (separators_node64[i])
                {
                    newNode->separators[k] = separators_node64[i];
                    currLvlPtrs[numNextLvlNodes] = &newNode->childsPtrs[k];
                    nextLvlBounds[numNextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                    numNextLvlNodes++;
                    k++;
                    startBound = bounds_node64[i];
                }
                else
                {
                    newNode->separators[g] = ULONG_MAX;
                    newNode->childsPtrs[g] = nullptr;
                    g++;
                }
            }

            *prevLvlPtrs[j] = (void *) newNode;

            fill(producedSeparators.begin(), producedSeparators.end(), 0);
            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
            fill(bounds_node64.begin(), bounds_node64.end(), 0);
        }

        prevLvlPtrs = move(currLvlPtrs);
        currLvlPtrs.clear();

        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();

        numNodesPerLevel = numNextLvlNodes;
    }

    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->dim = dim;
        newNode->type = Leaf;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            leafNodeSize = numOfElements;

            if (numOfElements > treeStats.maxLeafCapacity)
                treeStats.maxLeafCapacity = numOfElements;
        }

        types::Point<Dim> minCorner, maxCorner;

        for (i = 0; i < Dim; i++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [i](auto &a, auto &b){return a[i] < b[i];});
            minCorner[i] = (*minIt)[i];
            maxCorner[i] = (*maxIt)[i];
            newNode->records[i].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &p = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = p[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }

}

template<size_t Dim>
inline void collect_points_1way_nodes(void *root, types::Points<Dim> &points)
{
    uint8_t dim = 0, type;
    uint32_t metadata;
    stack<void*> stack;
    stack.push(root);
    while (!stack.empty())
    {
        void *node = stack.top();
        stack.pop();
        metadata = *reinterpret_cast<uint32_t *>(node);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *> (node);

                for (int i = actualNode->slotuse-1; i >= 0; i--)
                {
                    stack.push(actualNode->childsPtrs[i]);
                }
                break;
            }
            case NodeType::Leaf:
            {
                LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim> *>(node);
                uint32_t numRecords = currLeaf->slotuse;
                uint32_t offset = points.size();

                points.resize(numRecords + offset);
                for (uint32_t i = 0; i < numRecords; i++)
                {
                    types::Point<Dim> p;

                    for (uint32_t j = 0; j < Dim; j++)
                    {
                        p[j] = currLeaf->records[j][i];
                    }

                    points[i + offset] = p;
                }
                break;
            }
        }
    }
}

template <size_t Dim>
inline bool reconstruct_subtree_1way_nodes(void *root, uint32_t subtreeDepth, types::Point<Dim> point)
{
    uint8_t dim, type;
    uint32_t metadata = 0;
    types::Points<Dim> subtreePoints;

    subtreePoints.emplace_back(point);
    
    metadata = *reinterpret_cast<uint32_t *>(root);
    type = metadata & 0x3;
    dim = (metadata >> 2) & 0x3F;


    Node64 *actualRootNode = reinterpret_cast<Node64 *>(root);
    uint32_t pos = successorLinGNode64(actualRootNode->separators, (uint64_t) (point[dim] * CONVERSION_FACTOR));
    
    collect_points_1way_nodes(actualRootNode->childsPtrs[pos], subtreePoints);
    
    if (actualRootNode->slotuse == NODE_SIZE_64)
    {
        vector<uint8_t> subtreeDims;
        update_subtree_1way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
        return true;
    }

    uint32_t medianPos = subtreePoints.size() >> 1;            
    auto lowIt = subtreePoints.begin();
    auto medianIt = subtreePoints.begin() + medianPos;
    auto highIt = subtreePoints.end();

    nth_element(lowIt, medianIt, highIt, [dim](auto& a, auto& b){return a[dim] < b[dim];});

    uint64_t newSeparator = (uint64_t) (subtreePoints[medianPos][dim] * CONVERSION_FACTOR);
    double countingThreshold = subtreePoints[medianPos][dim];
    
    auto partitionIt = partition(lowIt, highIt, [countingThreshold, dim](auto& p) {
            return p[dim] < countingThreshold;
        }
    );

    uint32_t leftRegion = partitionIt - lowIt;
    uint32_t rightRegion = highIt - partitionIt;

    if (leftRegion < (leafCapacity >> 1))
    {
        vector<uint8_t> subtreeDims;
        update_subtree_1way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, subtreePoints.size(), subtreePoints, subtreeDims);
        return true;
    }

    uint32_t slotuse = actualRootNode->slotuse;
    uint32_t shiftedEntries = slotuse - pos;

    memmove(&actualRootNode->separators[pos+1], &actualRootNode->separators[pos], shiftedEntries * sizeof(uint64_t));
    actualRootNode->separators[pos] = newSeparator;

    void *newSubtree = actualRootNode->childsPtrs[slotuse];
    memmove(&actualRootNode->childsPtrs[pos+1], &actualRootNode->childsPtrs[pos], shiftedEntries * sizeof(void *));

    if (newSubtree == nullptr)
    {
        vector<uint8_t> subtreeDims;
        // update right subtree (i.e., ptr[pos+1])
        update_subtree_1way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
        
        // create left subtree (i.e., &ptr[pos])
        create_subtree_1way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
        
    }
    else
    {
    
        // update right subtree(i.e., ptr[pos+1])
        vector<uint8_t> subtreeDims;
        update_subtree_1way_nodes(actualRootNode->childsPtrs[pos+1], subtreeDepth+1, leftRegion, subtreePoints.size(), subtreePoints, subtreeDims);
    
        // update left subtree(i.e., ptr[pos])
        actualRootNode->childsPtrs[pos] = newSubtree;
        subtreeDims.clear();
        update_subtree_1way_nodes(actualRootNode->childsPtrs[pos], subtreeDepth+1, 0, leftRegion, subtreePoints, subtreeDims);
    }

    actualRootNode->slotuse++;  

    return true;
}

template <size_t Dim>
inline void make_split_1way_nodes(void *parentNode, LeafNode<Dim> *currLeaf, uint32_t parentMetadata, uint32_t pos)
{
    uint8_t type = parentMetadata & 0x3;
    uint8_t dim = (parentMetadata >> 2) & 0x3F;
    
    vector<uint32_t> indices(currLeaf->slotuse);
    iota(indices.begin(), indices.end(), 0);

    uint32_t medianPos = indices.size() >> 1;

    nth_element(indices.begin(), indices.begin() + medianPos, indices.end(),
                [&](uint32_t idx1, uint32_t idx2) {
                    return currLeaf->records[dim][idx1] < currLeaf->records[dim][idx2];
                }
    );

    uint64_t convertedElement = (uint64_t) (currLeaf->records[dim][indices[medianPos]] * CONVERSION_FACTOR);


    Node64 *actualParentNode = reinterpret_cast<Node64 *>(parentNode);
    uint64_t newSeparator = convertedElement;
    double countingThreshold = currLeaf->records[dim][indices[medianPos]];

    auto partitionIt = partition(indices.begin(), indices.end(),[&](uint32_t idx) {
                return currLeaf->records[dim][idx] < countingThreshold;
            }
    );

    size_t leftRegion = partitionIt - indices.begin();
    size_t rightRegion = indices.size() - leftRegion;

    if (leftRegion < (leafCapacity >> 1))
    {
        if (!currLeaf->is_outlier)
        {
            currLeaf->is_outlier = 1;
        }
        else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
            treeStats.maxLeafCapacity = currLeaf->slotuse;

        types::Point<Dim> minCorner, maxCorner;
        for (uint32_t j = 0; j < Dim; j++)
        {

            auto [minIt, maxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].end());
            minCorner[j] = *minIt;
            maxCorner[j] = *maxIt;
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }            

    void *newNode = nullptr;
    LeafNode<Dim> *newLeaf = nullptr;

    uint32_t slotuse = actualParentNode->slotuse;
    actualParentNode->slotuse++;
    uint32_t shiftedEntries = slotuse - pos;


    memmove(&actualParentNode->separators[pos+1], &actualParentNode->separators[pos], shiftedEntries * sizeof(uint64_t));
    actualParentNode->separators[pos] = newSeparator;
    
    newNode = actualParentNode->childsPtrs[slotuse];
    memmove(&actualParentNode->childsPtrs[pos+1], &actualParentNode->childsPtrs[pos], shiftedEntries * sizeof(void*));

    // There isn't an empty linked node from previous re-partitioning
    if (newNode == nullptr)
    {
        actualParentNode->childsPtrs[pos+1] = newNode = leafNodePool->allocate();
        newLeaf = new (newNode) LeafNode<Dim>();
    }
    else
    {
        actualParentNode->childsPtrs[pos+1] = newNode;
        newLeaf = new (newNode) LeafNode<Dim>();
        newLeaf->slotuse = 0;
        newLeaf->is_outlier = 0;

        for (uint32_t d = 0; d < Dim; d++)
        {
            newLeaf->records[d].clear();
        }
    }

    // Fill the vectors of each leaf node with the appropriate coordinates
    sort(indices.begin(), indices.begin() + leftRegion);

    for (uint32_t j = 0; j < Dim; j++)
    {
        newLeaf->records[j].resize(rightRegion);
        for (uint32_t i = 0; i < rightRegion; i++)
        {
            uint32_t idx = indices[leftRegion + i];
            newLeaf->records[j][i] = currLeaf->records[j][idx];
        }

        for (uint32_t i = 0; i < leftRegion; i++)
        {
            uint32_t idx = indices[i];
            currLeaf->records[j][i] = currLeaf->records[j][idx];
        }
    }

    currLeaf->slotuse = leftRegion;
    newLeaf->slotuse = rightRegion;
    
    if (leftRegion < treeStats.splitThreshold)
    {
        currLeaf->is_outlier = 0;
        leftRegion = treeStats.splitThreshold;
    }
    else if (leftRegion >= treeStats.splitThreshold && leftRegion < treeStats.outlierThreshold)
    {
        currLeaf->is_outlier = 0;
    }
    else
    {
        currLeaf->is_outlier = 1;
        treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, leftRegion);
    }
    
    if (rightRegion < treeStats.splitThreshold)
    {
        newLeaf->is_outlier = 0;
        rightRegion = treeStats.splitThreshold;
    }
    else if (rightRegion >= treeStats.splitThreshold && rightRegion < treeStats.outlierThreshold)
    {
        newLeaf->is_outlier = 0;
    }
    else
    {
        newLeaf->is_outlier = 1;
        treeStats.maxLeafCapacity = std::max(treeStats.maxLeafCapacity, rightRegion);
    }


    types::Point<Dim> oldMinCorner, oldMaxCorner;
    types::Point<Dim> newMinCorner, newMaxCorner;

    // Update the BBs and vector sizes
    for (uint32_t j = 0; j < Dim; j++)
    {
        currLeaf->records[j].resize(leftRegion);
        auto [oldMinIt, oldMaxIt] = minmax_element(currLeaf->records[j].begin(), currLeaf->records[j].begin() + currLeaf->slotuse);

        oldMinCorner[j] = *oldMinIt;
        oldMaxCorner[j] = *oldMaxIt;

        newLeaf->records[j].resize(rightRegion);
        auto [newMinIt, newMaxIt] = minmax_element(newLeaf->records[j].begin(), newLeaf->records[j].begin() + newLeaf->slotuse);

        newMinCorner[j] = *newMinIt;
        newMaxCorner[j] = *newMaxIt;
    }

    currLeaf->boundingBox = types::Box<Dim>(oldMinCorner, oldMaxCorner);

    newLeaf->dim = dim;
    newLeaf->type = Leaf;
    newLeaf->boundingBox = types::Box<Dim>(newMinCorner, newMaxCorner);

}

template <size_t Dim>
inline void insert_3way_nodes(types::Point<Dim> point)
{
    static uint32_t numUpdatesAtRoot = 0;
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, bool>> path(treeStats.treeDepth, {nullptr, false});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N16:
            {
                Node16 *actualNode = reinterpret_cast<Node16 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode16(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, (slotuse < NODE_SIZE_16)};            
                currNode = actualNode->childsPtrs[pos];

                break;
            }
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, (slotuse < NODE_SIZE_32)};
                currNode = actualNode->childsPtrs[pos];

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, (slotuse < NODE_SIZE_64)};
                currNode = actualNode->childsPtrs[pos];

                break;
            }
        }
    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);

    // Leaf node has an available entry for the insertion
    // Also BB must be updated
    if (currLeaf->slotuse < treeStats.splitThreshold)
    {
        uint32_t insertedPos = currLeaf->slotuse;
        currLeaf->slotuse++;

        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();

        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j][insertedPos] = point[j];

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }
    else if (!currLeaf->is_outlier && currLeaf->slotuse + 1 < treeStats.outlierThreshold)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;        
    }
    else if (currLeaf->is_outlier && currLeaf->slotuse + 1 < 2 * treeStats.maxLeafCapacity)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }

    auto& [parentNode, hasEmptySlot] = path[treeStats.treeDepth-1];

    if (hasEmptySlot)
    {
        currLeaf->slotuse++;
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);
        }

        make_split_3way_nodes(parentNode, currLeaf, metadata, pos);
        return;
    }    
    else
    {
        auto nodeWithSlot = pair<void*, bool> {nullptr, false};
        size_t depth = 0;
        for (int j = treeStats.treeDepth-2; j >= 0; j--)
        {
            if (path[j].second)
            {
                nodeWithSlot = path[j];
                depth = j;
                break;
            }
        }

        if (nodeWithSlot.second)
        {
            reconstruct_subtree_3way_nodes(nodeWithSlot.first, depth, point);
            return;
        }

        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        if (!currLeaf->is_outlier)
        {
            currLeaf->is_outlier = 1;
        }
        else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
        {
            treeStats.maxLeafCapacity = currLeaf->slotuse;
        }
    }    

}

template <size_t Dim>
inline void insert_2way_nodes(types::Point<Dim> point)
{
    static uint32_t numUpdatesAtRoot = 0;
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, bool>> path(treeStats.treeDepth, {nullptr, false});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, (slotuse < NODE_SIZE_32)};
                currNode = actualNode->childsPtrs[pos];

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, (slotuse < NODE_SIZE_64)};
                currNode = actualNode->childsPtrs[pos];

                break;
            }
        }
    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);

    // Leaf node has an available entry for the insertion
    // Also BB must be updated
    if (currLeaf->slotuse < treeStats.splitThreshold)
    {
        uint32_t insertedPos = currLeaf->slotuse;
        currLeaf->slotuse++;

        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();

        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j][insertedPos] = point[j];

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }
    else if (!currLeaf->is_outlier && currLeaf->slotuse + 1 < treeStats.outlierThreshold)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;        
    }
    else if (currLeaf->is_outlier && currLeaf->slotuse + 1 < 2 * treeStats.maxLeafCapacity)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }

    auto& [parentNode, hasEmptySlot] = path[treeStats.treeDepth-1];

    if (hasEmptySlot)
    {
        currLeaf->slotuse++;
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);
        }

        make_split_2way_nodes(parentNode, currLeaf, metadata, pos);
        return;
    }    
    else
    {
        auto nodeWithSlot = pair<void*, bool> {nullptr, false};
        size_t depth = 0;
        for (int j = treeStats.treeDepth-2; j >= 0; j--)
        {
            if (path[j].second)
            {
                nodeWithSlot = path[j];
                depth = j;
                break;
            }
        }

        if (nodeWithSlot.second)
        {
            reconstruct_subtree_2way_nodes(nodeWithSlot.first, depth, point);
            return;
        }

        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        if (!currLeaf->is_outlier)
        {
            currLeaf->is_outlier = 1;
        }
        else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
        {
            treeStats.maxLeafCapacity = currLeaf->slotuse;
        }
    }    

}

template <size_t Dim>
inline void insert_1way_nodes(types::Point<Dim> point)
{
    static uint32_t numUpdatesAtRoot = 0;
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, bool>> path(treeStats.treeDepth, {nullptr, false});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
        slotuse = actualNode->slotuse;
        pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

        path[i] = {currNode, (slotuse < NODE_SIZE_64)};
        currNode = actualNode->childsPtrs[pos];
    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);

    // Leaf node has an available entry for the insertion
    // Also BB must be updated
    if (currLeaf->slotuse < treeStats.splitThreshold)
    {
        uint32_t insertedPos = currLeaf->slotuse;
        currLeaf->slotuse++;

        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();

        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j][insertedPos] = point[j];

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }
    else if (!currLeaf->is_outlier && currLeaf->slotuse + 1 < treeStats.outlierThreshold)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;        
    }
    else if (currLeaf->is_outlier && currLeaf->slotuse + 1 < 2 * treeStats.maxLeafCapacity)
    {
        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        return;
    }

    auto& [parentNode, hasEmptySlot] = path[treeStats.treeDepth-1];

    if (hasEmptySlot)
    {
        currLeaf->slotuse++;
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);
        }

        make_split_1way_nodes(parentNode, currLeaf, metadata, pos);
        return;
    }    
    else
    {
        auto nodeWithSlot = pair<void*, bool> {nullptr, false};
        size_t depth = 0;
        for (int j = treeStats.treeDepth-2; j >= 0; j--)
        {
            if (path[j].second)
            {
                nodeWithSlot = path[j];
                depth = j;
                break;
            }
        }

        if (nodeWithSlot.second)
        {
            reconstruct_subtree_1way_nodes(nodeWithSlot.first, depth, point);
            return;
        }

        currLeaf->slotuse++;
        types::Point<Dim> minCorner, maxCorner;
        minCorner = currLeaf->boundingBox.min_corner();
        maxCorner = currLeaf->boundingBox.max_corner();
        for (uint32_t j = 0; j < Dim; j++)
        {
            currLeaf->records[j].emplace_back(point[j]);

            if (point[j] < minCorner[j])
                minCorner[j] = point[j];

            if(point[j] > maxCorner[j])
                maxCorner[j] = point[j];
        }

        currLeaf->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        if (!currLeaf->is_outlier)
        {
            currLeaf->is_outlier = 1;
        }
        else if (currLeaf->slotuse > treeStats.maxLeafCapacity)
        {
            treeStats.maxLeafCapacity = currLeaf->slotuse;
        }
    }    

}


template <size_t Dim>
using insert_func = void(*) (types::Point<Dim>);

template <size_t Dim>
inline insert_func<Dim> insertPoint[] = {
    insert_3way_nodes<Dim>,
    insert_2way_nodes<Dim>, 
    insert_1way_nodes<Dim> 
};


/*
 * skd-tree Removal Functions
 * --------------------------
 * These routines remove a point from an skd-tree, supporting different
 * internal node configurations:
 * 
 * 1. remove_3way_nodes   → N16, N32, N64 internal nodes allowed
 * 2. remove_2way_nodes   → N32, N64 internal nodes allowed
 * 3. remove_1way_nodes   → only N64 internal nodes allowed
 *
 * Removal Procedure:
 * ------------------
 * 1. **Tree Traversal**:
 *    - Traverse from the root down to the leaf corresponding to the
 *      converted coordinates of the point.
 *    - At each level, the appropriate successor function
 *      (`successorLinGNode16/32/64`) is used to locate the child node
 *      containing the point along the current split dimension.
 *    - The path is recorded along with the position in the parent node
 *      for potential updates to separators and child pointers.
 *
 * 2. **Leaf Removal**:
 *    - Verify that the point lies within the leaf's bounding box.
 *    - Use SIMD-accelerated masks to locate the exact position of the
 *      point in the leaf records.
 *    - Shift remaining entries to fill the gap left by the removed point.
 *    - Update the leaf's `slotuse` counter and resize record arrays
 *      if necessary.
 *
 * 3. **Leaf Bounding Box Update**:
 *    - After deletion, recalculate the minimum and maximum coordinates
 *      of the leaf to maintain correct bounding boxes.
 *    - If the leaf becomes empty, update the parent nodes along the
 *      recorded path, shifting separators and child pointers as needed.
 *
 * 4. **Tree Maintenance**:
 *    - Adjust the `is_outlier` flag if the leaf drops below the outlier
 *      threshold.
 *    - Ensure that parent node separators correctly reflect the reduced
 *      key range after deletion.
 *
 * The `removePoint` table provides function pointers for dispatching
 * deletions based on the selected tree configuration, avoiding runtime
 * branching in the hot removal path.
 *
 * Notes:
 *  - All three removal routines share a common path traversal and leaf
 *    handling logic; only the internal node types differ.
 *  - Bounding boxes are always updated to maintain correctness of future queries.
 *  - Outlier handling ensures that temporary over-capacity leaves do not
 *    interfere with deletion correctness.
 */

template <size_t Dim>
inline void remove_3way_nodes(types::Point<Dim> point)
{
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, size_t>> path(treeStats.treeDepth, {nullptr, 0});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N16:
            {
                Node16 *actualNode = reinterpret_cast<Node16 *>(currNode);
                pos = successorLinGNode16(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, pos};   
                currNode = actualNode->childsPtrs[pos];
                break;
            }
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, pos};   
                currNode = actualNode->childsPtrs[pos];

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, pos};   
                currNode = actualNode->childsPtrs[pos];
                break;
            }
        }
    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);
    
    if (currLeaf->slotuse == 0)
        return;
    types::Point<Dim> leafMinCoords = currLeaf->boundingBox.min_corner(), leafMaxCoords = currLeaf->boundingBox.max_corner();


    for (int d = 0; d < Dim; d++)
    {
        if (point[d] < leafMinCoords[d] || point[d] > leafMaxCoords[d])
            return;
    }

    uint32_t numRecords = currLeaf->slotuse; 
    uint32_t numBlocks = numRecords >> 3;
    uint32_t remainder = numRecords & 7; 
    uint32_t activeBlocks = numBlocks;
    vector<__mmask8> masks(numBlocks, 0xFF);

    for (int d = 0; d < Dim && activeBlocks; d++)
    {
        __m512d q_vec = _mm512_set1_pd(point[d]);

        for (int b = 0; b < numBlocks; b++)
        {
            if (!masks[b])
                continue;
            
            size_t offset = b << 3;
            __m512d coords_vec = _mm512_loadu_pd(currLeaf->records[d].data() + offset);

            masks[b] &= _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!masks[b])
                activeBlocks--;
        }
    }

    for (int b = 0; b < numBlocks && activeBlocks; b++)
    {
        if (!masks[b])
            continue;
        
        size_t offset = (b << 3) + _tzcnt_u16(masks[b]);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }    

        goto UPDATE_LEAF;
    }

    if (remainder)
    {
        __mmask8 tailMask = (__mmask8) (1u << remainder) - 1;
        __mmask8 validMask = tailMask;
        size_t offset = numBlocks << 3;

        for (int d = 0; d < Dim; d++)
        {
            __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, currLeaf->records[d].data() + offset);
            __m512d q_vec = _mm512_set1_pd(point[d]);
            validMask &=  _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!validMask)
                return;
        }

        offset += _tzcnt_u16(validMask);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }

    }


UPDATE_LEAF:

    if (currLeaf->slotuse == 0)
    {
        for (int i = treeStats.treeDepth - 1; i >= 0; i--)
        {
            currNode = path[i].first;
            pos = path[i].second;
            metadata = *reinterpret_cast<uint32_t *>(currNode);
            type = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;
            
            switch (type)
            {
                case NodeType::N16:
                {
                    Node16 *actualNode = reinterpret_cast<Node16 *>(currNode);

                    if (actualNode->slotuse == 1)
                    {
                        actualNode->slotuse--;
                        break;
                    }

                    if (pos == NODE_SIZE_16 - 1)
                    {
                        actualNode->slotuse--;
                        actualNode->separators[pos - 1] = USHRT_MAX;
                        
                        return;
                    }
                    else
                    {
                        void *ptr = actualNode->childsPtrs[pos];
                        size_t movedEntries = actualNode->slotuse - pos - 1;
                        memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint16_t));
                        memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                        actualNode->slotuse--;
                        
                        actualNode->separators[actualNode->slotuse] = USHRT_MAX;
                        actualNode->childsPtrs[actualNode->slotuse] = ptr;
                        
                        return;
                    }

                    break;
                }
                case NodeType::N32:
                {
                    Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);

                    if (actualNode->slotuse == 1)
                    {
                        actualNode->slotuse--;
                        break;
                    }

                    if (pos == NODE_SIZE_32 - 1)
                    {
                        actualNode->slotuse--;
                        actualNode->separators[pos - 1] = UINT_MAX;
                        return;
                    }
                    else
                    {
                        void *ptr = actualNode->childsPtrs[pos];
                        size_t movedEntries = actualNode->slotuse - pos - 1;
                        memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint32_t));
                        memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                        actualNode->slotuse--;
                        
                        actualNode->separators[actualNode->slotuse] = UINT_MAX;
                        actualNode->childsPtrs[actualNode->slotuse] = ptr;
                        
                        return;
                    }

                    break;
                }
                case NodeType::N64:
                {
                    Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);

                    if (actualNode->slotuse == 1)
                    {
                        actualNode->slotuse--;
                        break;
                    }

                    if (pos == NODE_SIZE_64 - 1)
                    {
                        actualNode->slotuse--;
                        actualNode->separators[pos - 1] = ULONG_MAX;
                        return;
                    }
                    else
                    {
                        void *ptr = actualNode->childsPtrs[pos];
                        size_t movedEntries = actualNode->slotuse - pos - 1;
                        memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint64_t));
                        memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                        actualNode->slotuse--;
                        
                        actualNode->separators[actualNode->slotuse] = ULONG_MAX;
                        actualNode->childsPtrs[actualNode->slotuse] = ptr;
                        
                        return;
                    }

                    break;
                }
            }
        }       
    }

    for (int d = 0; d < Dim; d++)
    {
        auto [minIt, maxIt] = minmax_element(currLeaf->records[d].begin(), currLeaf->records[d].begin() + currLeaf->slotuse);
        leafMinCoords[d] = *minIt;
        leafMaxCoords[d] = *maxIt;
    }

    currLeaf->boundingBox = types::Box<Dim>(leafMinCoords, leafMaxCoords);

    if (currLeaf->slotuse < treeStats.outlierThreshold)
        currLeaf->is_outlier = 0;
}

template <size_t Dim>
inline void remove_2way_nodes(types::Point<Dim> point)
{
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, size_t>> path(treeStats.treeDepth, {nullptr, 0});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        switch (type)
        {
            case NodeType::N32:
            {
                Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode32(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, pos};   
                currNode = actualNode->childsPtrs[pos];

                break;
            }
            case NodeType::N64:
            {
                Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
                slotuse = actualNode->slotuse;
                pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

                path[i] = {currNode, pos};   
                currNode = actualNode->childsPtrs[pos];
                break;
            }
        }
    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);
    
    if (currLeaf->slotuse == 0)
        return;
    types::Point<Dim> leafMinCoords = currLeaf->boundingBox.min_corner(), leafMaxCoords = currLeaf->boundingBox.max_corner();


    for (int d = 0; d < Dim; d++)
    {
        if (point[d] < leafMinCoords[d] || point[d] > leafMaxCoords[d])
            return;
    }

    uint32_t numRecords = currLeaf->slotuse; 
    uint32_t numBlocks = numRecords >> 3;
    uint32_t remainder = numRecords & 7; 
    uint32_t activeBlocks = numBlocks;
    vector<__mmask8> masks(numBlocks, 0xFF);

    for (int d = 0; d < Dim && activeBlocks; d++)
    {
        __m512d q_vec = _mm512_set1_pd(point[d]);

        for (int b = 0; b < numBlocks; b++)
        {
            if (!masks[b])
                continue;
            
            size_t offset = b << 3;
            __m512d coords_vec = _mm512_loadu_pd(currLeaf->records[d].data() + offset);

            masks[b] &= _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!masks[b])
                activeBlocks--;
        }
    }

    for (int b = 0; b < numBlocks && activeBlocks; b++)
    {
        if (!masks[b])
            continue;
        
        size_t offset = (b << 3) + _tzcnt_u16(masks[b]);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }


        goto UPDATE_LEAF;
    }

    if (remainder)
    {
        __mmask8 tailMask = (__mmask8) (1u << remainder) - 1;
        __mmask8 validMask = tailMask;
        size_t offset = numBlocks << 3;

        for (int d = 0; d < Dim; d++)
        {
            __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, currLeaf->records[d].data() + offset);
            __m512d q_vec = _mm512_set1_pd(point[d]);
            validMask &=  _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!validMask)
                return;
        }

        offset += _tzcnt_u16(validMask);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }

    }


UPDATE_LEAF:

    if (currLeaf->slotuse == 0)
    {
        for (int i = treeStats.treeDepth - 1; i >= 0; i--)
        {
            currNode = path[i].first;
            pos = path[i].second;
            metadata = *reinterpret_cast<uint32_t *>(currNode);
            type = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;
            
            switch (type)
            {
                case NodeType::N32:
                {
                    Node32 *actualNode = reinterpret_cast<Node32 *>(currNode);

                    if (actualNode->slotuse == 1)
                    {
                        actualNode->slotuse--;
                        break;
                    }

                    if (pos == NODE_SIZE_32 - 1)
                    {
                        actualNode->slotuse--;
                        actualNode->separators[pos - 1] = UINT_MAX;
                        return;
                    }
                    else
                    {
                        void *ptr = actualNode->childsPtrs[pos];
                        size_t movedEntries = actualNode->slotuse - pos - 1;
                        memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint32_t));
                        memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                        actualNode->slotuse--;
                        
                        actualNode->separators[actualNode->slotuse] = UINT_MAX;
                        actualNode->childsPtrs[actualNode->slotuse] = ptr;
                        
                        return;
                    }

                    break;
                }
                case NodeType::N64:
                {
                    Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);

                    if (actualNode->slotuse == 1)
                    {
                        actualNode->slotuse--;
                        break;
                    }

                    if (pos == NODE_SIZE_64 - 1)
                    {
                        actualNode->slotuse--;
                        actualNode->separators[pos - 1] = ULONG_MAX;
                        return;
                    }
                    else
                    {
                        void *ptr = actualNode->childsPtrs[pos];
                        size_t movedEntries = actualNode->slotuse - pos - 1;
                        memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint64_t));
                        memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                        actualNode->slotuse--;
                        
                        actualNode->separators[actualNode->slotuse] = ULONG_MAX;
                        actualNode->childsPtrs[actualNode->slotuse] = ptr;
                        
                        return;
                    }

                    break;
                }
            }
        }       
    }

    for (int d = 0; d < Dim; d++)
    {
        auto [minIt, maxIt] = minmax_element(currLeaf->records[d].begin(), currLeaf->records[d].begin() + currLeaf->slotuse);
        leafMinCoords[d] = *minIt;
        leafMaxCoords[d] = *maxIt;
    }

    currLeaf->boundingBox = types::Box<Dim>(leafMinCoords, leafMaxCoords);

    if (currLeaf->slotuse < treeStats.outlierThreshold)
        currLeaf->is_outlier = 0;
}

template <size_t Dim>
inline void remove_1way_nodes(types::Point<Dim> point)
{
    uint8_t dim = 0, type;
    uint32_t metadata, pos = 0, slotuse;
    uint64_t convertedCoordinates[Dim];
    void *currNode = root;
    vector<pair<void *, size_t>> path(treeStats.treeDepth, {nullptr, 0});

    for (uint32_t i = 0; i < Dim; i++)
    {
        convertedCoordinates[i] = (uint64_t) (point[i] * CONVERSION_FACTOR);
    }

    for (uint32_t i = 0; i < treeStats.treeDepth; i++)
    {
        metadata = *reinterpret_cast<uint32_t *>(currNode);
        type = metadata & 0x3;
        dim = (metadata >> 2) & 0x3F;

        Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);
        slotuse = actualNode->slotuse;
        pos = successorLinGNode64(actualNode->separators, convertedCoordinates[dim]);

        path[i] = {currNode, pos};   
        currNode = actualNode->childsPtrs[pos];

    }

    LeafNode<Dim> *currLeaf = reinterpret_cast<LeafNode<Dim>*>(currNode);
    
    if (currLeaf->slotuse == 0)
        return;
    types::Point<Dim> leafMinCoords = currLeaf->boundingBox.min_corner(), leafMaxCoords = currLeaf->boundingBox.max_corner();


    for (int d = 0; d < Dim; d++)
    {
        if (point[d] < leafMinCoords[d] || point[d] > leafMaxCoords[d])
            return;
    }

    uint32_t numRecords = currLeaf->slotuse; 
    uint32_t numBlocks = numRecords >> 3;
    uint32_t remainder = numRecords & 7; 
    uint32_t activeBlocks = numBlocks;
    vector<__mmask8> masks(numBlocks, 0xFF);

    for (int d = 0; d < Dim && activeBlocks; d++)
    {
        __m512d q_vec = _mm512_set1_pd(point[d]);

        for (int b = 0; b < numBlocks; b++)
        {
            if (!masks[b])
                continue;
            
            size_t offset = b << 3;
            __m512d coords_vec = _mm512_loadu_pd(currLeaf->records[d].data() + offset);

            masks[b] &= _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!masks[b])
                activeBlocks--;
        }
    }

    for (int b = 0; b < numBlocks && activeBlocks; b++)
    {
        if (!masks[b])
            continue;
        
        size_t offset = (b << 3) + _tzcnt_u16(masks[b]);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }

        goto UPDATE_LEAF;
    }

    if (remainder)
    {
        __mmask8 tailMask = (__mmask8) (1u << remainder) - 1;
        __mmask8 validMask = tailMask;
        size_t offset = numBlocks << 3;

        for (int d = 0; d < Dim; d++)
        {
            __m512d coords_vec = _mm512_maskz_loadu_pd(tailMask, currLeaf->records[d].data() + offset);
            __m512d q_vec = _mm512_set1_pd(point[d]);
            validMask &=  _mm512_cmp_pd_mask(coords_vec, q_vec, _CMP_EQ_OQ);

            if (!validMask)
                return;
        }

        offset += _tzcnt_u16(validMask);

        if (offset != numRecords - 1)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d][offset] = currLeaf->records[d][numRecords-1];
            }
        }

        currLeaf->slotuse--;
        if (currLeaf->slotuse >= treeStats.splitThreshold)
        {
            for (int d = 0; d < Dim; d++)
            {
                currLeaf->records[d].resize(currLeaf->slotuse);
            }   
        }

    }


UPDATE_LEAF:

    if (currLeaf->slotuse == 0)
    {
        for (int i = treeStats.treeDepth - 1; i >= 0; i--)
        {
            currNode = path[i].first;
            pos = path[i].second;
            metadata = *reinterpret_cast<uint32_t *>(currNode);
            type = metadata & 0x3;
            dim = (metadata >> 2) & 0x3F;
            
            Node64 *actualNode = reinterpret_cast<Node64 *>(currNode);

            if (actualNode->slotuse == 1)
            {
                actualNode->slotuse--;
                break;
            }

            if (pos == NODE_SIZE_64 - 1)
            {
                actualNode->slotuse--;
                actualNode->separators[pos - 1] = ULONG_MAX;
                return;
            }
            else
            {
                void *ptr = actualNode->childsPtrs[pos];
                size_t movedEntries = actualNode->slotuse - pos - 1;
                memmove(&actualNode->separators[pos], &actualNode->separators[pos+1], movedEntries * sizeof(uint64_t));
                memmove(&actualNode->childsPtrs[pos], &actualNode->childsPtrs[pos+1], movedEntries * sizeof(void*));
                actualNode->slotuse--;
                
                actualNode->separators[actualNode->slotuse] = ULONG_MAX;
                actualNode->childsPtrs[actualNode->slotuse] = ptr;
                
                return;
            }

        }       
    }

    for (int d = 0; d < Dim; d++)
    {
        auto [minIt, maxIt] = minmax_element(currLeaf->records[d].begin(), currLeaf->records[d].begin() + currLeaf->slotuse);
        leafMinCoords[d] = *minIt;
        leafMaxCoords[d] = *maxIt;
    }

    currLeaf->boundingBox = types::Box<Dim>(leafMinCoords, leafMaxCoords);

    if (currLeaf->slotuse < treeStats.outlierThreshold)
        currLeaf->is_outlier = 0;
}

template <size_t Dim>
using remove_func = void(*) (types::Point<Dim>);

template <size_t Dim>
inline remove_func<Dim> removePoint[] = {
    remove_3way_nodes<Dim>,
    remove_2way_nodes<Dim>, 
    remove_1way_nodes<Dim> 
};


