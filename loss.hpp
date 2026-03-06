#pragma once
#include "opencl_runtime.hpp"
#include "tensor.hpp"
#include <vector>

class Loss{
public:
    virtual float forward(const Tensor& predictions, const Tensor& targets, size_t batch_size) = 0;
    virtual void backward(const Tensor& predictions, const Tensor& targets, Tensor& grad_inputs, size_t batch_size) = 0;
};

class MSE : public Loss{
private:
    size_t size;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int> forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int> backward_kernel;
public:
    MSE(size_t size_);
    float forward(const Tensor& predictions, const Tensor& targets, size_t batch_size) override;
    void backward(const Tensor& predictions, const Tensor& targets, Tensor& grad_inputs, size_t batch_size) override;
};
class CrossEntropy : public Loss{
private:
    size_t size;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int> forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int,int> backward_kernel;
public:
    CrossEntropy(size_t size_);
    float forward(const Tensor& predictions, const Tensor& targets, size_t batch_size) override;
    void backward(const Tensor& predictions, const Tensor& targets, Tensor& grad_inputs, size_t batch_size) override;
};