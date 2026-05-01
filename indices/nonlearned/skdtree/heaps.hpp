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

namespace heaps{
template<size_t Dim>
class maxHeapNode{
    using Point = point_t<Dim>;

    public :
        double dist;
        Point p;

        maxHeapNode(){}
        maxHeapNode(double dist, Point p)
        {
            this->dist = dist;
            this->p = p;
        }


};

template<size_t Dim>
class maxHeap : public std::vector<maxHeapNode<Dim>>{
    public:
        uint32_t currCapacity, capacity;

        maxHeap(uint32_t k)
        {
            this->currCapacity = 0;
            this->capacity = k;
            this->reserve(this->capacity);
        }

        ~maxHeap(){}
        // void compute();
        void enheap(maxHeapNode<Dim> node)
        {
            int lastPos = this->size();
            int p;
            maxHeapNode<Dim> tmp;

            this->push_back(node);
            
            while (lastPos > 0){
                p = parent(lastPos);

                if (node.dist < this->at(p).dist){
                    break;
                }
                else{
                    tmp = this->at(p);
                    this->at(p) = node;
                    this->at(lastPos) = tmp;
                    lastPos = parent(lastPos);
                }

            }
        }

        uint32_t parent(uint32_t position)
        {
            if (position%2){
                return position/2;
            }
            else{
                return (position-1)/2;
            }
        }
        void movedown()
        {
            int pos = 0;
            int swap;
            maxHeapNode<Dim> tmp;

            while (pos*2 + 1 < this->size()){
                if (pos*2 + 2 < this->size()){
                    if (this->at(pos*2+1).dist < this->at(pos*2+2).dist ){
                        swap = pos*2+2;
                    }
                    else{
                        
                        swap = pos*2+1;
                    }
                }
                else{
                    swap = pos*2+1;
                }

                if (this->at(pos).dist > this->at(swap).dist){
                break;
                }
                else{
                    tmp = this->at(swap);
                    this->at(swap) = this->at(pos);
                    this->at(pos) = tmp;

                    pos = swap;
                }
            }
        }
        void insert(maxHeapNode<Dim> node)
        {
            if (currCapacity < capacity){     
                enheap(node);
                currCapacity++;
            }   
            else{
                if (node.dist > this->at(0).dist){
                    return;
                }
                else{
                    this->at(0) = node;
                    movedown();
                }
            }            
        }
        maxHeapNode<Dim> deheap()
        {
            maxHeapNode<Dim> node;
            node = this->at(this->currCapacity-1);
            this->pop_back();
            this->currCapacity--;

            return node;
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