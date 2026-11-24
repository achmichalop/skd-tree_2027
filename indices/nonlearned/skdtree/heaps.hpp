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
#include <cstddef>
#include "tree_core.hpp"

namespace heaps{
template<size_t Dim>
class maxHeapNode{
    using LeafNode = Leaf_Node<Dim>;

    public :
        uint32_t idx;
        double dist;
        LeafNode *leaf;

        maxHeapNode(){}
        maxHeapNode(double dist, LeafNode *leaf, uint32_t idx)
        {
            this->idx = idx;
            this->dist = dist;
            this->leaf = leaf;
        }
};

template<size_t Dim>
class maxHeap
{
public:
    std::vector<maxHeapNode<Dim>> data;
    uint32_t capacity;

    explicit maxHeap(uint32_t k)
        : capacity(k)
    {
        data.reserve(k);
    }


    inline const maxHeapNode<Dim>& top() const
    {
        return data[0];
    }

    inline size_t size() const
    {
        return data.size();
    }

    inline void siftUp(uint32_t i)
    {
        auto node = std::move(data[i]);

        while (i > 0)
        {
            uint32_t p = (i - 1) >> 1;

            if (data[p].dist >= node.dist)
                break;

            data[i] = std::move(data[p]);
            i = p;
        }

        data[i] = std::move(node);
    }

    inline void siftDown(uint32_t i)
    {
        uint32_t n = data.size();
        auto node = std::move(data[i]);

        while (true)
        {
            uint32_t l = (i << 1) + 1;
            if (l >= n) break;

            uint32_t r = l + 1;

            uint32_t best = (r < n && data[r].dist > data[l].dist) ? r : l;

            if (data[best].dist <= node.dist)
                break;

            data[i] = std::move(data[best]);
            i = best;
        }

        data[i] = std::move(node);
    }

    inline void insert(const maxHeapNode<Dim> node)
    {
        if (data.size() < capacity)
        {
            data.push_back(node);
            siftUp(data.size() - 1);
        }
        else if (node.dist < data[0].dist)
        {
            data[0] = node;
            siftDown(0);
        }
    }
};


enum MIN_HEAP_TYPES
{
    SINGLE_NODE = 0,
    LEFT_GROUP,
    RIGHT_GROUP
};

template<size_t Dim>
class minHeapNode
{
    public: 
        int8_t idx, type;
        void* node;
        double dist;
        std::array<double, Dim> projection_dists;

        minHeapNode(){}
        minHeapNode(void *node, int8_t idx, int8_t type, double dist, const std::array<double, Dim> &projection_dists)
        {
            this->idx = idx;
            this->type = type;
            this->node = node;
            this->dist = dist;
            this->projection_dists = projection_dists;
        }

        bool operator>(const minHeapNode& other) const {
            return dist > other.dist;
        }
};

template<size_t Dim>
class minHeap : public std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>
{        
    public:

        using std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>::push;
        using std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>::top;
        using std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>::pop;
        using std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>::empty;
        using std::priority_queue<minHeapNode<Dim>, std::vector<minHeapNode<Dim>>, std::greater<minHeapNode<Dim>>>::size;
};
}