#pragma once
#include <vector>
#include "opencl_runtime.hpp"

class Tensor {
    private:
        mutable std::vector<float> data;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        mutable cl::Buffer buffer;
        mutable bool on_gpu;
        size_t total_size;
        size_t sizebytes;

        void calculateStrides();

        inline size_t getFlatIndex(const std::vector<size_t>& indices);


    public: 
        Tensor(const std::vector<size_t>& shape_);
        Tensor(const std::vector<size_t>& shape_, const std::vector<float>& data_);
        ~Tensor();
        size_t getTotalSize();

        void toGPU() const;
        void toCPU() const;
        void toGPU();
        void toCPU();

        std::vector<float> getData() const;
        std::vector<size_t> getShape() const;
        cl::Buffer getBuffer();
        cl::Buffer getBuffer() const;
        std::vector<float>& getDataRef();

        void set(const std::vector<size_t>& indices, float value);
        float get(const std::vector<size_t>& indices);


        void print(const size_t max_elements = 10);
};

void combine_tensors(const std::vector<Tensor*>& tensors, Tensor& output);