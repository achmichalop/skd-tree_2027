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
#include "dimRanking.hpp"
#include "tree_core.hpp"

using namespace std;
using namespace chrono;

size_t totalPoints = 0;
size_t totalLeaves = 0;
size_t heavyLeaves = 0;
size_t outlierLeaves = 0;


/*
 * Separator Extraction Utilities
 * ------------------------------
 * This set of helper functions is responsible for extracting the desired number
 * of separators (i.e., splitters) for internal nodes during skd-tree construction.
 *
 * The extraction process is specialized for each internal node layout (N16, N32,
 * and N64) and adapts the separator representation to the corresponding level of
 * key compression. For compressed layouts (N16 and N32), separators are derived
 * from quantized prefixes of coordinate values, whereas the baseline layout (N64)
 * uses exact 64-bit separator values.
 *
 * Each extraction routine operates on a subset of points and computes candidate
 * split positions using partial selection (nth_element) and partitioning, while
 * enforcing minimum occupancy constraints to avoid highly unbalanced splits.
 * The resulting separators and partition boundaries define the child subranges
 * of the internal node.
 *
 * The verifyNode() function is used only when multiple node layouts are evaluated
 * during tree construction. In this case, it validates and compacts the produced
 * separators and bounds for the finally selected layout, ensuring consistency
 * with the chosen representation. When a single layout is used, this verification
 * step is skipped.
 */


template<size_t Dim>
void extractSeparatorsNode16(types::Points<Dim> &points, uint32_t currentDim, vector <uint64_t> &separators, uint32_t expectedSeparators, vector <uint32_t> &bounds, uint32_t startBoundPos, uint32_t endBoundPos)
{
    uint32_t lowerBoundPos = 0;
    uint32_t upperBoundPos = 0;
    uint32_t depth = ceil(log2((double)expectedSeparators));
    uint64_t mask = ((uint64_t) USHRT_MAX << NODE_16_TRAILING_ZEROS);
    uint32_t step = ceil((endBoundPos-startBoundPos) / (double) expectedSeparators);
    uint32_t thr = (step > leafCapacity) ? step >> 1 : leafCapacity >> 1;
    bounds[0] = startBoundPos;
    fill(bounds.begin() + expectedSeparators - 1, bounds.end(), endBoundPos);

    for (uint32_t d = depth; d > 0; --d)
    {
        uint32_t low = 0;
        uint32_t stride = 1 << d;
        uint32_t high = stride - 1;
        uint32_t numIterations = 1 << (NODE_DEPTH_16 - d);

        for (uint32_t j = 0; j < numIterations; ++j)
        {
            lowerBoundPos = bounds[low];
            upperBoundPos = bounds[high];
            uint32_t examinedSepPos = low + ((high - low) >> 1);

            if (examinedSepPos >= (expectedSeparators - 1))
                break;


            uint32_t medianElemPos = startBoundPos + (examinedSepPos + 1) * step;

            if (!separators[high] || medianElemPos >= (upperBoundPos - thr))
            {
                separators[examinedSepPos] = 0;
                bounds[examinedSepPos] = medianElemPos;
                low = high;
                high += stride;               
                continue;
            }

            auto lowIt = points.begin() + lowerBoundPos;
            auto medianIt = points.begin() + medianElemPos;
            auto highIt = points.begin() + upperBoundPos;

            nth_element(lowIt, medianIt, highIt, [currentDim](auto& a, auto& b){return a[currentDim] < b[currentDim];});

            uint64_t examinedSeparator = ((uint64_t) (points[medianElemPos][currentDim] * CONVERSION_FACTOR)) & mask;
            double countingThreshold = examinedSeparator / (double) CONVERSION_FACTOR;
            
            auto partitionIt = partition(lowIt, highIt, [countingThreshold, currentDim](auto& p){return p[currentDim] < countingThreshold;});

            uint32_t counter = partitionIt - points.begin();
            
            if ((counter - lowerBoundPos) < thr)
            {
                examinedSeparator = 0;
                counter = medianElemPos;
            }

            separators[examinedSepPos] = examinedSeparator;
            bounds[examinedSepPos] = counter;
            low = high;
            high += stride;
        }
    }

    fill(separators.begin() + expectedSeparators, separators.end(), 0);
}

template<size_t Dim>
void extractSeparatorsNode32(types::Points<Dim> &points, uint32_t currentDim, vector <uint64_t> &separators, uint32_t expectedSeparators, vector <uint32_t> &bounds, uint32_t startBoundPos, uint32_t endBoundPos)
{
    uint32_t lowerBoundPos = 0;
    uint32_t upperBoundPos = 0;
    uint32_t depth = ceil(log2((double) expectedSeparators));
    uint64_t mask = ((uint64_t) UINT_MAX << NODE_32_TRAILING_ZEROS);
    uint32_t step = ceil((endBoundPos-startBoundPos) / (double) expectedSeparators);
    uint32_t thr = (step > leafCapacity) ? step >> 1 : leafCapacity >> 1;

    bounds[0] = startBoundPos;

    fill(bounds.begin() + expectedSeparators - 1, bounds.end(), endBoundPos);
    
    // Find the desired number of separators for this node
    for (uint32_t d = depth; d > 0; --d)
    {
        uint32_t low = 0;
        uint32_t stride = 1 << d;
        uint32_t high = stride - 1;
        uint32_t numIterations = 1 << (NODE_DEPTH_32 - d);

        for (uint32_t j = 0; j < numIterations; ++j)
        {
            lowerBoundPos = bounds[low];
            upperBoundPos = bounds[high];
            uint32_t examinedSepPos = low + ((high - low) >> 1);
    
            if (examinedSepPos >= (expectedSeparators - 1))
                break;
            
            uint32_t medianElemPos = startBoundPos + (examinedSepPos + 1) * step;

            if (!separators[high] || medianElemPos >= (upperBoundPos - thr))
            {
                separators[examinedSepPos] = 0;
                bounds[examinedSepPos] = medianElemPos;
                low = high;
                high += stride;
                continue;
            }

            auto lowIt = points.begin() + lowerBoundPos;
            auto medianIt = points.begin() + medianElemPos;
            auto highIt = points.begin() + upperBoundPos;

            nth_element(lowIt, medianIt, highIt, [currentDim](auto& a, auto& b){return a[currentDim] < b[currentDim];});

            uint64_t examinedSeparator = ((uint64_t) (points[medianElemPos][currentDim] * CONVERSION_FACTOR)) & mask;
            double countingThreshold = examinedSeparator / (double) CONVERSION_FACTOR;

            auto partitionIt = partition(lowIt, highIt, [countingThreshold, currentDim](auto& p){return p[currentDim] < countingThreshold;});

            uint32_t counter = partitionIt - points.begin();

            if ((counter - lowerBoundPos) < thr)
            {
                examinedSeparator = 0;
                counter = medianElemPos;
            }

            separators[examinedSepPos] = examinedSeparator;
            bounds[examinedSepPos] = counter;

            low = high;
            high += stride;

        }
    }

    fill(separators.begin() + expectedSeparators, separators.end(), 0);
}

template<size_t Dim>
void extractSeparatorsNode64(types::Points<Dim> &points, uint32_t currentDim, vector <uint64_t> &separators, uint32_t expectedSeparators, vector <uint32_t> &bounds, uint32_t startBoundPos, uint32_t endBoundPos)
{
    uint32_t lowerBoundPos = 0;
    uint32_t upperBoundPos = 0;
    uint32_t depth = ceil(log2((double) expectedSeparators));

    uint32_t step = ceil((endBoundPos-startBoundPos) / (double) expectedSeparators);
    uint32_t thr = (step > leafCapacity) ? step >> 1 : leafCapacity >> 1;

    bounds[0] = startBoundPos;
    fill(bounds.begin() + expectedSeparators - 1, bounds.end(), endBoundPos);

    for (uint32_t d = depth; d > 0; d--)
    {
        uint32_t low = 0;
        uint32_t stride = 1 << d;
        uint32_t high = stride - 1;
        uint32_t numIterations = 1 << (NODE_DEPTH_64 - d);
        
        for (uint32_t j = 0; j < numIterations; j++)
        {
            lowerBoundPos = bounds[low];
            upperBoundPos = bounds[high];
            uint32_t examinedSepPos = low + ((high - low) >> 1);

            if (examinedSepPos >= (expectedSeparators - 1))
                break;

            uint32_t medianElemPos = startBoundPos + (examinedSepPos + 1) * step;
            
            if (!separators[high] || medianElemPos >= (upperBoundPos - thr))
            {
                separators[examinedSepPos] = 0;
                bounds[examinedSepPos] = medianElemPos;
                low = high;
                high += stride;
                continue;
            }

            auto lowIt = points.begin() + lowerBoundPos;
            auto medianIt = points.begin() + medianElemPos;
            auto highIt = points.begin() + upperBoundPos;

            nth_element(lowIt, medianIt, highIt, [currentDim](auto& a, auto& b){return a[currentDim] < b[currentDim];});
    
            uint64_t examinedSeparator = ((uint64_t) (points[medianElemPos][currentDim] * CONVERSION_FACTOR));
            double countingThreshold = points[medianElemPos][currentDim];

            auto partitionIt = partition(lowIt, highIt, [countingThreshold, currentDim](auto& p){return p[currentDim] < countingThreshold;});

            uint32_t counter = partitionIt - points.begin();

            if ((counter - lowerBoundPos) < thr)
            {
                examinedSeparator = 0;
                counter = medianElemPos;
            }

            separators[examinedSepPos] = examinedSeparator;
            bounds[examinedSepPos] = counter;

            low = high;
            high += stride;
        }
    }

    fill(separators.begin() + expectedSeparators, separators.end(), 0);

}

template<size_t Dim>
void verifyNode(types::Points<Dim> &points, uint32_t currentDim, uint32_t producedSeparators, vector <uint64_t> &separators, vector <uint32_t> &bounds, uint32_t startBoundPos, uint32_t endBoundPos)
{
    uint32_t lowerBoundPos = startBoundPos;
    uint32_t upperBoundPos = endBoundPos;

    for (uint32_t i = 0, k = 0; k < producedSeparators-1; i++)
    {
        if (separators[i])
        {
            double countingThreshold = separators[i] / (double) CONVERSION_FACTOR;

            auto lowIt = points.begin() + lowerBoundPos;
            auto medianIt = points.begin() + bounds[i];
            auto highIt = points.begin() + upperBoundPos;
            nth_element(lowIt, medianIt, highIt, [currentDim](auto& a, auto& b){return a[currentDim] < b[currentDim];});
            auto partitionIt = partition(lowIt, highIt, [countingThreshold, currentDim](auto& p){return p[currentDim] < countingThreshold;});
            
            bounds[k] = partitionIt - points.begin();
            separators[k] = separators[i];
            lowerBoundPos = bounds[k];
            k++;
        }
    }

    separators[producedSeparators - 1] = ULONG_MAX;
    bounds[producedSeparators - 1] = upperBoundPos;
}


/*
 * skd-tree Construction Routines
 * ------------------------------
 * This set of functions implements the construction logic of the skd-tree.
 * Each routine corresponds to a specific tree configuration and determines
 * which internal node layouts (N16, N32, and/or N64) are enabled during index
 * construction.
 *
 *  - buildIndex_3way_nodes(): enables all three internal node layouts and
 *    dynamically selects between N16, N32, and N64 at each level based on
 *    the number of produced separators and local split characteristics.
 *
 *  - buildIndex_2way_nodes(): enables a restricted configuration where only
 *    the N32 and N64 internal node layouts are considered.
 *
 *  - buildIndex_1way_nodes(): constructs the baseline tree using only the
 *    uncompressed N64 internal node layout.
 *
 * The construction proceeds in a level-by-level (breadth-first) manner.
 * At each level, a dimension is selected according to a specified order,
 * candidate separators are extracted, and the most suitable node layout is
 * chosen while enforcing minimum occupancy constraints to avoid degenerate
 * splits. The process terminates when subspaces reach leaf-level granularity,
 * after which leaf nodes are created.
 *
 * A table of function pointers is used to dispatch the construction call to
 * the appropriate routine based on the tree type selected in tree_core.hpp.
 * This design avoids runtime branching inside the hot construction path and
 * cleanly separates layout-specific construction logic.
 */

template<size_t Dim>
void buildIndex_3way_nodes(types::Points<Dim> &points)
{
    uint32_t examinedDims = 0, numOfElements = 0;
    uint32_t i = 0, j = 0, k = 0, g = 0, nodeType;
    uint32_t numNodesPerLevel = 1, nextLvlNodes = 0;
    uint32_t startBound = 0, endBound = points.size();
    uint32_t level = 0, currentDim = 0, expectedSeparators = 0;
    uint32_t avgProducedSeparatorsPerLvl = 0;

    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node16(NODE_SIZE_16, 0), bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node16(NODE_SIZE_16, ULONG_MAX), separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;
    

    ranker::DimRanker<Dim> ranker(points);
    uint32_t maxDepth = Dim;
    double minSubspace = DBL_MAX, maxSubspace = 0.0f;
    double meanPerLevel = 0.0f, sumPerLevel = 0.0f, sumSqPerLevel = 0.0f;
    double stddevPerLevel = 0.0f;

    currLvlBounds.push_back(make_pair(0,endBound));
    prevLvlPtrs.push_back(&root);

    /* Using a sample of the dataset to make the dimensionality ranking */
    auto rankedDims = ranker.rankDimensions(0.1);

    while (numNodesPerLevel)
    {

        nextLvlNodes = 0;
        currLvlPtrs.resize(numNodesPerLevel * NODE_SIZE_16);
        nextLvlBounds.resize(numNodesPerLevel * NODE_SIZE_16);
        currentDim = rankedDims[examinedDims].dimension;
        
        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            uint32_t subspaceLeaves = (uint32_t) round(numOfElements / (double) leafCapacity);
            uint32_t localSplits;

            if (level < Dim)
            {
                uint32_t exponent = Dim - examinedDims;
                localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
            }
            else
            {
                if (subspaceLeaves <= NODE_SIZE_64)
                {
                    localSplits = subspaceLeaves;
                }
                else
                {
                    uint32_t exponent = maxDepth - level;
                    localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
                }
            }


            if (localSplits == 1)
            {
                Node64 *newNode = static_cast<Node64 *> (internalNodePool->allocate());
                newNode->dim = currentDim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;
                
                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void*) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [currentDim](auto &a, auto&b){return a[currentDim] < b[currentDim];});   
            if ((*minIt)[currentDim] == (*maxIt)[currentDim])
            {
                Node64 *newNode = static_cast<Node64 *> (internalNodePool->allocate());
                newNode->dim = currentDim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;
                
                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void*) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }

            if (localSplits > NODE_SIZE_32)
            {
                nodeType = N16;
            }
            else if (localSplits > NODE_SIZE_64)
            {
                nodeType = N32;
            }
            else
            {
                nodeType = N64;
            }


            switch (nodeType)
            {
                case N16:
                {   

                    expectedSeparators = (localSplits >= NODE_SIZE_16) ? NODE_SIZE_16 : localSplits;
                    extractSeparatorsNode16(points, currentDim, separators_node16, expectedSeparators, bounds_node16, startBound, endBound);
                    producedSeparators[0] = NODE_SIZE_16 - count(separators_node16.begin(), separators_node16.end(), 0);

                    if (producedSeparators[0] >= NODE_SIZE_32)
                    {

                        Node16 *newNode = static_cast<Node16 *>(internalNodePool->allocate());
                        newNode->slotuse = producedSeparators[0];
                        newNode->dim = currentDim;
                        newNode->type = N16;

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_16; i++)
                        {
                            if (separators_node16[i])
                            {
                                newNode->separators[k] = (uint16_t) (separators_node16[i] >> NODE_16_TRAILING_ZEROS);
                                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node16[i]);
                                
                                double subspaceSize = (double) (bounds_node16[i] - startBound);
                                sumPerLevel += subspaceSize;
                                sumSqPerLevel += subspaceSize * subspaceSize;
                            
                                if (subspaceSize < minSubspace)
                                    minSubspace = subspaceSize;
                                
                                if (subspaceSize > maxSubspace)
                                    maxSubspace = subspaceSize;
                                
                                k++;
                                nextLvlNodes++;
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
                        
                        avgProducedSeparatorsPerLvl += newNode->slotuse;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node16.begin(), separators_node16.end(), ULONG_MAX);
                        fill(bounds_node16.begin(), bounds_node16.end(), 0);
                        break;
                    }
                    else
                    {
                        expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                        extractSeparatorsNode32(points, currentDim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                        producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                        if (producedSeparators[1] > producedSeparators[0])
                        {
                            Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                            newNode->slotuse = producedSeparators[1];
                            newNode->dim = currentDim;
                            newNode->type = N32;

                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                            {
                                if (separators_node32[i])
                                {
                                    newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                    currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                    
                                    double subspaceSize = (double) (bounds_node32[i] - startBound);
                                    sumPerLevel += subspaceSize;
                                    sumSqPerLevel += subspaceSize * subspaceSize;
                                
                                    if (subspaceSize < minSubspace)
                                        minSubspace = subspaceSize;
                                    
                                    if (subspaceSize > maxSubspace)
                                        maxSubspace = subspaceSize;
                                    
                                    k++;
                                    nextLvlNodes++;
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

                            avgProducedSeparatorsPerLvl += newNode->slotuse;

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
                            extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                            producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                            if (producedSeparators[2] > producedSeparators[0])
                            {

                                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                                newNode->slotuse = producedSeparators[2];
                                newNode->dim = currentDim;
                                newNode->type = N64;

                                for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                                {
                                    if (separators_node64[i])
                                    {

                                        newNode->separators[k] = separators_node64[i];
                                        currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                        nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                        double subspaceSize = (double) (bounds_node64[i] - startBound);
                                        sumPerLevel += subspaceSize;
                                        sumSqPerLevel += subspaceSize * subspaceSize;
                                    
                                        if (subspaceSize < minSubspace)
                                            minSubspace = subspaceSize;
                                        
                                        if (subspaceSize > maxSubspace)
                                            maxSubspace = subspaceSize;
                                        
                                        k++;
                                        nextLvlNodes++;
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

                                avgProducedSeparatorsPerLvl += newNode->slotuse;

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

                    verifyNode(points, currentDim, producedSeparators[0], separators_node16, bounds_node16, startBound, endBound);

                    Node16 *newNode = static_cast<Node16 *>(internalNodePool->allocate());
                    newNode->slotuse = producedSeparators[0];
                    newNode->dim = currentDim;
                    newNode->type = N16;


                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint16_t) (separators_node16[i] >> NODE_16_TRAILING_ZEROS);
                        currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node16[i]);
                        
                        double subspaceSize = (double) (bounds_node16[i] - startBound);
                        sumPerLevel += subspaceSize;
                        sumSqPerLevel += subspaceSize * subspaceSize;
                    
                        if (subspaceSize < minSubspace)
                            minSubspace = subspaceSize;
                        
                        if (subspaceSize > maxSubspace)
                            maxSubspace = subspaceSize;

                        nextLvlNodes++;
                        startBound = bounds_node16[i];
                    }

                    for (; i < NODE_SIZE_16; i++)
                    {
                        newNode->separators[i] = USHRT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;                    
                    avgProducedSeparatorsPerLvl += newNode->slotuse;

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
                    expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                    extractSeparatorsNode32(points, currentDim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    if (producedSeparators[1] >= NODE_SIZE_64)
                    {
                        Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                        newNode->slotuse = producedSeparators[1];
                        newNode->dim = currentDim;
                        newNode->type = N32;

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            if (separators_node32[i])
                            {
                                newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                
                                double subspaceSize = (double) (bounds_node32[i] - startBound);
                                sumPerLevel += subspaceSize;
                                sumSqPerLevel += subspaceSize * subspaceSize;
                            
                                if (subspaceSize < minSubspace)
                                    minSubspace = subspaceSize;
                                
                                if (subspaceSize > maxSubspace)
                                    maxSubspace = subspaceSize;
                                
                                k++;
                                nextLvlNodes++;
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

                        avgProducedSeparatorsPerLvl += newNode->slotuse;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                        fill(bounds_node32.begin(), bounds_node32.end(), 0);
                        break;
                    }
                    else
                    {
                        expectedSeparators = NODE_SIZE_64;
                        extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                        producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);
                        
                        if (producedSeparators[2] > producedSeparators[1])
                        {
                            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                            newNode->slotuse = producedSeparators[2];
                            newNode->dim = currentDim;
                            newNode->type = N64;
                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                            {
                                if (separators_node64[i])
                                {
                                    newNode->separators[k] = separators_node64[i];
                                    currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                    
                                    double subspaceSize = (double) (bounds_node64[i] - startBound);
                                    sumPerLevel += subspaceSize;
                                    sumSqPerLevel += subspaceSize * subspaceSize;
                                
                                    if (subspaceSize < minSubspace)
                                        minSubspace = subspaceSize;
                                    
                                    if (subspaceSize > maxSubspace)
                                        maxSubspace = subspaceSize;
                                                                        
                                    k++;
                                    nextLvlNodes++;
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
                            avgProducedSeparatorsPerLvl += newNode->slotuse;
                            
                            fill(producedSeparators.begin(), producedSeparators.end(), 0);
                            fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                            fill(bounds_node32.begin(), bounds_node32.end(), 0);
                            fill(bounds_node64.begin(), bounds_node64.end(), 0);
                            break;
                        }
                    }


                    verifyNode(points, currentDim, producedSeparators[1], separators_node32, bounds_node32, startBound, endBound);

                    Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                    newNode->slotuse = producedSeparators[1];
                    newNode->dim = currentDim;
                    newNode->type = N32;

                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                        currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                        
                        double subspaceSize = (double) (bounds_node32[i] - startBound);
                        sumPerLevel += subspaceSize;
                        sumSqPerLevel += subspaceSize * subspaceSize;
                    
                        if (subspaceSize < minSubspace)
                            minSubspace = subspaceSize;
                        
                        if (subspaceSize > maxSubspace)
                            maxSubspace = subspaceSize;

                        nextLvlNodes++;
                        startBound = bounds_node32[i];

                    }

                    for (; i < NODE_SIZE_32; i++)
                    {
                        newNode->separators[i] = UINT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;
                    avgProducedSeparatorsPerLvl += newNode->slotuse;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);

                    break;
                }
                case N64:
                {
                    
                    expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
                    extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                    newNode->slotuse = producedSeparators[2];
                    newNode->dim = currentDim;
                    newNode->type = N64;

                    for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            newNode->separators[k] = separators_node64[i];
                            currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                            nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                            
                            double subspaceSize = (double) (bounds_node64[i] - startBound);
                            sumPerLevel += subspaceSize;
                            sumSqPerLevel += subspaceSize * subspaceSize;
                        
                            if (subspaceSize < minSubspace)
                                minSubspace = subspaceSize;
                            
                            if (subspaceSize > maxSubspace)
                                maxSubspace = subspaceSize;

                            k++;
                            nextLvlNodes++;
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

                    avgProducedSeparatorsPerLvl += newNode->slotuse;

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
        
        avgProducedSeparatorsPerLvl = round(avgProducedSeparatorsPerLvl / (double) numNodesPerLevel);

        meanPerLevel = sumPerLevel / nextLvlNodes;
        double variance = (sumSqPerLevel / nextLvlNodes) - (meanPerLevel * meanPerLevel);
        stddevPerLevel = sqrt(variance);

        if (round(meanPerLevel/ (double) leafCapacity) <= 1)
        {
            numNodesPerLevel = nextLvlNodes;
            level++;

            examinedDims++;
            if (examinedDims == Dim)
                examinedDims = 0;
            
            treeStats.splitThreshold = 1.2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.outlierThreshold = 2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.treeDepth = maxDepth;
            treeStats.maxLeafCapacity = max((size_t)maxSubspace, treeStats.outlierThreshold);

            totalLeaves = numNodesPerLevel;
            totalPoints = points.size();

            break;
        }

        avgProducedSeparatorsPerLvl = 0;
        numNodesPerLevel = nextLvlNodes;
        level++;

        if (level == Dim)
        {
            maxDepth += ceil(log2(maxSubspace / (double) leafCapacity) / (double) NODE_DEPTH_16);
        }

        examinedDims++;
        if (examinedDims == Dim)
            examinedDims = 0;

        sumPerLevel = 0.0f;
        sumSqPerLevel = 0.0f;
        minSubspace = DBL_MAX;
        maxSubspace = 0.0f;
    }


    //Construct leaf nodes
    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->type = Leaf;
        newNode->dim = currentDim;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            heavyLeaves++;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            outlierLeaves++;
            leafNodeSize = numOfElements;
        }
        
        types::Point<Dim> minCorner, maxCorner;
        
        for (uint32_t j = 0; j < Dim; j++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [j](auto &a, auto &b){return a[j] < b[j];});
            minCorner[j] = (*minIt)[j];
            maxCorner[j] = (*maxIt)[j];

            newNode->records[j].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &r = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = r[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }

}

template<size_t Dim>
void buildIndex_2way_nodes(types::Points<Dim> &points)
{
    uint32_t examinedDims = 0, numOfElements = 0;
    uint32_t i = 0, j = 0, k = 0, g = 0, nodeType;
    uint32_t numNodesPerLevel = 1, nextLvlNodes = 0;
    uint32_t startBound = 0, endBound = points.size();
    uint32_t level = 0, currentDim = 0, expectedSeparators = 0;
    uint32_t avgProducedSeparatorsPerLvl = 0;

    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node32(NODE_SIZE_32, 0), bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node32(NODE_SIZE_32, ULONG_MAX), separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;
    
    ranker::DimRanker<Dim> ranker(points);
    uint32_t maxDepth = Dim;
    double minSubspace = DBL_MAX, maxSubspace = 0.0f;
    double meanPerLevel = 0.0f, sumPerLevel = 0.0f, sumSqPerLevel = 0.0f;
    double stddevPerLevel = 0.0f;


    currLvlBounds.push_back(make_pair(0,endBound));
    prevLvlPtrs.push_back(&root);

    /* Using a sample of the dataset to make the dimensionality ranking */
    auto rankedDims = ranker.rankDimensions(0.1);

    while (numNodesPerLevel)
    {

        nextLvlNodes = 0;
        currLvlPtrs.resize(numNodesPerLevel * NODE_SIZE_32);
        nextLvlBounds.resize(numNodesPerLevel * NODE_SIZE_32);
        currentDim = rankedDims[examinedDims].dimension;
        
        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            uint32_t subspaceLeaves = (uint32_t) round(numOfElements / (double) leafCapacity);

            uint32_t localSplits;

            if (level < Dim)
            {
                uint32_t exponent = Dim - examinedDims;
                localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
            }
            else
            {
                if (subspaceLeaves <= NODE_SIZE_64)
                {
                    localSplits = subspaceLeaves;
                }
                else
                {
                    uint32_t exponent = maxDepth - level;
                    localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
                }
            }

            if (localSplits == 1)
            {
                Node64 *newNode = static_cast<Node64 *> (internalNodePool->allocate());
                newNode->dim = currentDim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;

                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void*) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [currentDim](auto &a, auto&b){return a[currentDim] < b[currentDim];});                       
            if ((*minIt)[currentDim] == (*maxIt)[currentDim])
            {
                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                newNode->slotuse = 1;
                newNode->dim = currentDim;
                newNode->type = N64;
                
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;

                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void *) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }


            if (localSplits > NODE_SIZE_64)
            {
                nodeType = N32;
            }
            else
            {
                nodeType = N64;
            }
            
            switch (nodeType)
            {
                case N32:
                {
                    expectedSeparators = (localSplits >= NODE_SIZE_32) ? NODE_SIZE_32 : localSplits;
                    extractSeparatorsNode32(points, currentDim, separators_node32, expectedSeparators, bounds_node32, startBound, endBound);
                    producedSeparators[1] = NODE_SIZE_32 - count(separators_node32.begin(), separators_node32.end(), 0);

                    if (producedSeparators[1] >= NODE_SIZE_64)
                    {
                        Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                        newNode->slotuse = producedSeparators[1];
                        newNode->dim = currentDim;
                        newNode->type = N32;

                        for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_32; i++)
                        {
                            if (separators_node32[i])
                            {
                                newNode->separators[k] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                                
                                double subspaceSize = (double) (bounds_node32[i] - startBound);
                                sumPerLevel += subspaceSize;
                                sumSqPerLevel += subspaceSize * subspaceSize;
                            
                                if (subspaceSize < minSubspace)
                                    minSubspace = subspaceSize;
                                
                                if (subspaceSize > maxSubspace)
                                    maxSubspace = subspaceSize;
                                
                                k++;
                                nextLvlNodes++;
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

                        avgProducedSeparatorsPerLvl += newNode->slotuse;

                        fill(producedSeparators.begin(), producedSeparators.end(), 0);
                        fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                        fill(bounds_node32.begin(), bounds_node32.end(), 0);
                        break;
                    }
                    else
                    {
                        expectedSeparators = NODE_SIZE_64;
                        extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                        producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);
                        
                        if (producedSeparators[2] > producedSeparators[1])
                        {
                            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                            newNode->slotuse = producedSeparators[2];
                            newNode->dim = currentDim;
                            newNode->type = N64;
                            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                            {
                                if (separators_node64[i])
                                {
                                    newNode->separators[k] = separators_node64[i];
                                    currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                                    nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                                    
                                    double subspaceSize = (double) (bounds_node64[i] - startBound);
                                    sumPerLevel += subspaceSize;
                                    sumSqPerLevel += subspaceSize * subspaceSize;
                                
                                    if (subspaceSize < minSubspace)
                                        minSubspace = subspaceSize;
                                    
                                    if (subspaceSize > maxSubspace)
                                        maxSubspace = subspaceSize;
                                    
                                    // cout << "\tsep " << k << ": " << separators_node64[i] << " -- [" << startBound << ", " << bounds_node64[i] << ")" << endl; 
                                    
                                    k++;
                                    nextLvlNodes++;
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
                            avgProducedSeparatorsPerLvl += newNode->slotuse;
                            
                            fill(producedSeparators.begin(), producedSeparators.end(), 0);
                            fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                            fill(bounds_node32.begin(), bounds_node32.end(), 0);
                            fill(bounds_node64.begin(), bounds_node64.end(), 0);
                            break;
                        }
                    }

                    verifyNode(points, currentDim, producedSeparators[1], separators_node32, bounds_node32, startBound, endBound);

                    Node32 *newNode = static_cast<Node32 *>(internalNodePool->allocate());
                    newNode->slotuse = producedSeparators[1];
                    newNode->dim = currentDim;
                    newNode->type = N32;

                    for (i = 0; i < newNode->slotuse; i++)
                    {
                        newNode->separators[i] = (uint32_t) (separators_node32[i] >> NODE_32_TRAILING_ZEROS);
                        currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[i];
                        nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node32[i]);
                        
                        double subspaceSize = (double) (bounds_node32[i] - startBound);
                        sumPerLevel += subspaceSize;
                        sumSqPerLevel += subspaceSize * subspaceSize;
                    
                        if (subspaceSize < minSubspace)
                            minSubspace = subspaceSize;
                        
                        if (subspaceSize > maxSubspace)
                            maxSubspace = subspaceSize;

                        nextLvlNodes++;
                        startBound = bounds_node32[i];

                    }

                    for (; i < NODE_SIZE_32; i++)
                    {
                        newNode->separators[i] = UINT_MAX;
                        newNode->childsPtrs[i] = nullptr;
                    }

                    *prevLvlPtrs[j] = (void *) newNode;
                    avgProducedSeparatorsPerLvl += newNode->slotuse;

                    fill(producedSeparators.begin(), producedSeparators.end(), 0);
                    fill(separators_node32.begin(), separators_node32.end(), ULONG_MAX);
                    fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
                    fill(bounds_node32.begin(), bounds_node32.end(), 0);
                    fill(bounds_node64.begin(), bounds_node64.end(), 0);

                    break;
                }
                case N64:
                {                    
                    expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
                    extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
                    producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

                    Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                    newNode->slotuse = producedSeparators[2];
                    newNode->dim = currentDim;
                    newNode->type = N64;

                    for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
                    {
                        if (separators_node64[i])
                        {
                            newNode->separators[k] = separators_node64[i];
                            currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                            nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                            
                            double subspaceSize = (double) (bounds_node64[i] - startBound);
                            sumPerLevel += subspaceSize;
                            sumSqPerLevel += subspaceSize * subspaceSize;
                        
                            if (subspaceSize < minSubspace)
                                minSubspace = subspaceSize;
                            
                            if (subspaceSize > maxSubspace)
                                maxSubspace = subspaceSize;

                            k++;
                            nextLvlNodes++;
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

                    avgProducedSeparatorsPerLvl += newNode->slotuse;

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
        
        avgProducedSeparatorsPerLvl = round(avgProducedSeparatorsPerLvl / (double) numNodesPerLevel);

        meanPerLevel = sumPerLevel / nextLvlNodes;
        double variance = (sumSqPerLevel / nextLvlNodes) - (meanPerLevel * meanPerLevel);
        stddevPerLevel = sqrt(variance);

        if (round(meanPerLevel/ (double) leafCapacity) <= 1)
        {
            numNodesPerLevel = nextLvlNodes;
            level++;

            examinedDims++;
            if (examinedDims == Dim)
                examinedDims = 0;

            treeStats.splitThreshold = 1.2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.outlierThreshold = 2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.treeDepth = maxDepth;
            treeStats.maxLeafCapacity = max((size_t)maxSubspace, treeStats.outlierThreshold);
            totalPoints = points.size();
            totalLeaves = numNodesPerLevel;

            break;
        }

        avgProducedSeparatorsPerLvl = 0;
        numNodesPerLevel = nextLvlNodes;
        level++;

        if (level == Dim)
        {
            maxDepth += ceil(log2(maxSubspace / (double) leafCapacity) / (double) NODE_DEPTH_32);
        }

        examinedDims++;
        if (examinedDims == Dim)
            examinedDims = 0;

        sumPerLevel = 0.0f;
        sumSqPerLevel = 0.0f;
        minSubspace = DBL_MAX;
        maxSubspace = 0.0f;
    }

    //Construct leaf nodes
    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->type = Leaf;
        newNode->dim = currentDim;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            heavyLeaves++;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            outlierLeaves++;
            leafNodeSize = numOfElements;
        }
        
        types::Point<Dim> minCorner, maxCorner;
        
        for (uint32_t j = 0; j < Dim; j++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [j](auto &a, auto &b){return a[j] < b[j];});
            minCorner[j] = (*minIt)[j];
            maxCorner[j] = (*maxIt)[j];

            newNode->records[j].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &r = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = r[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }
}

template<size_t Dim>
void buildIndex_1way_nodes(types::Points<Dim> &points)
{
    uint32_t examinedDims = 0, numOfElements = 0;
    uint32_t i = 0, j = 0, k = 0, g = 0, nodeType;
    uint32_t numNodesPerLevel = 1, nextLvlNodes = 0;
    uint32_t startBound = 0, endBound = points.size();
    uint32_t level = 0, currentDim = 0, expectedSeparators = 0;
    uint32_t avgProducedSeparatorsPerLvl = 0;

    vector <uint32_t> producedSeparators(3, 0);
    vector<uint32_t> bounds_node64(NODE_SIZE_64, 0);
    vector<uint64_t> separators_node64(NODE_SIZE_64, ULONG_MAX);    
    vector <void **> prevLvlPtrs, currLvlPtrs;
    vector <pair<uint32_t, uint32_t>> currLvlBounds, nextLvlBounds;
    
    ranker::DimRanker<Dim> ranker(points);

    uint32_t maxDepth = Dim;
    double minSubspace = DBL_MAX, maxSubspace = 0.0f;
    double meanPerLevel = 0.0f, sumPerLevel = 0.0f, sumSqPerLevel = 0.0f;
    double stddevPerLevel = 0.0f;

    currLvlBounds.push_back(make_pair(0,endBound));
    prevLvlPtrs.push_back(&root);

    /* Using a sample of the dataset to make the dimensionality ranking */
    auto rankedDims = ranker.rankDimensions(0.1);

    while (numNodesPerLevel)
    {

        nextLvlNodes = 0;
        currLvlPtrs.resize(numNodesPerLevel * NODE_SIZE_64);
        nextLvlBounds.resize(numNodesPerLevel * NODE_SIZE_64);
        currentDim = rankedDims[examinedDims].dimension;
        
        for (j = 0; j < numNodesPerLevel; j++)
        {
            startBound = currLvlBounds[j].first;
            endBound = currLvlBounds[j].second;
            numOfElements = endBound - startBound;

            uint32_t subspaceLeaves = (uint32_t) round(numOfElements / (double) leafCapacity);
            
            uint32_t localSplits;

            if (level < Dim)
            {
                uint32_t exponent = Dim - examinedDims;
                localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
            }
            else
            {
                if (subspaceLeaves <= NODE_SIZE_64)
                {
                    localSplits = subspaceLeaves;
                }
                else
                {
                    uint32_t exponent = maxDepth - level;
                    localSplits = ceil(pow(subspaceLeaves, 1.0f/exponent));
                }
            }

            if (localSplits == 1)
            {
                Node64 *newNode = static_cast<Node64 *> (internalNodePool->allocate());
                newNode->dim = currentDim;
                newNode->type = N64;
                newNode->slotuse = 1;
                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;
                
                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void*) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }

            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [currentDim](auto &a, auto&b){return a[currentDim] < b[currentDim];});
            if ((*minIt)[currentDim] == (*maxIt)[currentDim])
            {
                Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
                newNode->slotuse = 1;
                newNode->dim = currentDim;
                newNode->type = N64;

                newNode->separators[0] = ULONG_MAX;
                currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[0];
                nextLvlBounds[nextLvlNodes] = make_pair(startBound, endBound);
                nextLvlNodes++;
                
                double subspaceSize = numOfElements;
                sumPerLevel += subspaceSize;
                sumSqPerLevel += subspaceSize * subspaceSize;
            
                if (subspaceSize < minSubspace)
                    minSubspace = subspaceSize;
                
                if (subspaceSize > maxSubspace)
                    maxSubspace = subspaceSize;

                for (i = newNode->slotuse; i < NODE_SIZE_64; i++)
                {
                    newNode->separators[i] = ULONG_MAX;
                    newNode->childsPtrs[i] = nullptr;
                }
                
                *prevLvlPtrs[j] = (void *) newNode;

                avgProducedSeparatorsPerLvl += newNode->slotuse;

                continue;
            }
            
            expectedSeparators = (localSplits >= NODE_SIZE_64) ? NODE_SIZE_64 : localSplits;
            extractSeparatorsNode64(points, currentDim, separators_node64, expectedSeparators, bounds_node64, startBound, endBound);
            producedSeparators[2] = NODE_SIZE_64 - count(separators_node64.begin(), separators_node64.end(), 0);

            Node64 *newNode = static_cast<Node64 *>(internalNodePool->allocate());
            newNode->slotuse = producedSeparators[2];
            newNode->dim = currentDim;
            newNode->type = N64;

            for (i = 0, k = 0, g = newNode->slotuse; i < NODE_SIZE_64; i++)
            {
                if (separators_node64[i])
                {
                    newNode->separators[k] = separators_node64[i];
                    currLvlPtrs[nextLvlNodes] = &newNode->childsPtrs[k];
                    nextLvlBounds[nextLvlNodes] = make_pair(startBound, bounds_node64[i]);
                    
                    double subspaceSize = (double) (bounds_node64[i] - startBound);
                    sumPerLevel += subspaceSize;
                    sumSqPerLevel += subspaceSize * subspaceSize;
                
                    if (subspaceSize < minSubspace)
                        minSubspace = subspaceSize;
                    
                    if (subspaceSize > maxSubspace)
                        maxSubspace = subspaceSize;

                    k++;
                    nextLvlNodes++;
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

            avgProducedSeparatorsPerLvl += newNode->slotuse;

            fill(producedSeparators.begin(), producedSeparators.end(), 0);
            fill(separators_node64.begin(), separators_node64.end(), ULONG_MAX);
            fill(bounds_node64.begin(), bounds_node64.end(), 0);
        }

        prevLvlPtrs = move(currLvlPtrs);
        currLvlPtrs.clear();

        currLvlBounds = move(nextLvlBounds);
        nextLvlBounds.clear();
        
        avgProducedSeparatorsPerLvl = round(avgProducedSeparatorsPerLvl / (double) numNodesPerLevel);

        meanPerLevel = sumPerLevel / nextLvlNodes;
        double variance = (sumSqPerLevel / nextLvlNodes) - (meanPerLevel * meanPerLevel);
        stddevPerLevel = sqrt(variance);

        if (round(meanPerLevel/ (double) leafCapacity) <= 1)
        {
            numNodesPerLevel = nextLvlNodes;
            level++;

            examinedDims++;
            if (examinedDims == Dim)
                examinedDims = 0;

            treeStats.splitThreshold = 1.2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.outlierThreshold = 2 * (max(leafCapacity, (size_t)round(meanPerLevel)));
            treeStats.treeDepth = maxDepth;
            treeStats.maxLeafCapacity = max((size_t)maxSubspace, treeStats.outlierThreshold);
            totalPoints = points.size();
            totalLeaves = numNodesPerLevel;
            break;
        }

        avgProducedSeparatorsPerLvl = 0;
        numNodesPerLevel = nextLvlNodes;
        level++;

        if (level == Dim)
        {
            maxDepth += ceil(log2(maxSubspace / (double) leafCapacity) / (double) NODE_DEPTH_64);
        }

        examinedDims++;
        if (examinedDims == Dim)
            examinedDims = 0;

        sumPerLevel = 0.0f;
        sumSqPerLevel = 0.0f;
        minSubspace = DBL_MAX;
        maxSubspace = 0.0f;
    }

    //Construct leaf nodes
    for (j = 0; j < numNodesPerLevel; j++)
    {
        startBound = currLvlBounds[j].first;
        endBound = currLvlBounds[j].second;
        numOfElements = endBound - startBound;
        size_t leafNodeSize = 0;

        LeafNode<Dim> *newNode = new (leafNodePool->allocate()) LeafNode<Dim>();
        newNode->type = Leaf;
        newNode->dim = currentDim;
        newNode->slotuse = numOfElements;

        if (numOfElements <= treeStats.splitThreshold)
        {
            newNode->is_outlier = 0;
            leafNodeSize = treeStats.splitThreshold;
        }
        else if (numOfElements > treeStats.splitThreshold && numOfElements < treeStats.outlierThreshold)
        {
            newNode->is_outlier = 0;
            heavyLeaves++;
            leafNodeSize = numOfElements;
        }
        else
        {
            newNode->is_outlier = 1;
            outlierLeaves++;
            leafNodeSize = numOfElements;
        }
        
        types::Point<Dim> minCorner, maxCorner;
        
        for (uint32_t j = 0; j < Dim; j++)
        {
            auto [minIt, maxIt] = minmax_element(points.begin() + startBound, points.begin() + endBound, [j](auto &a, auto &b){return a[j] < b[j];});
            minCorner[j] = (*minIt)[j];
            maxCorner[j] = (*maxIt)[j];

            newNode->records[j].resize(leafNodeSize);
        }

        newNode->boundingBox = types::Box<Dim>(minCorner, maxCorner);

        for (i = startBound, k = 0; i < endBound; i++, k++)
        {
            auto &r = points[i];
            for (uint32_t j = 0; j < Dim; j++)
            {
                newNode->records[j][k] = r[j];
            }
        }

        *prevLvlPtrs[j] = (void *) newNode;
    }
}

template <size_t Dim>
using BuildIndexFunc = void(*)(types::Points<Dim>&);

template <size_t Dim>
inline BuildIndexFunc<Dim> buildIndex[] = {
    buildIndex_3way_nodes<Dim>, 
    buildIndex_2way_nodes<Dim>, 
    buildIndex_1way_nodes<Dim>  
};