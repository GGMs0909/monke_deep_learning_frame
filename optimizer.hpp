#pragma once
#include "opencl_runtime.hpp"
#include "tensor.hpp"

class Optimizer {
public:
    virtual void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters,float learning_rate) = 0;
    virtual void initialize_momentum(std::vector<Tensor*> parameters) {};
};

class GD : public Optimizer {
private:
    cl::make_kernel<cl::Buffer, cl::Buffer, float, int> gd_update_kernel;
    cl::make_kernel<cl::Buffer, int> zero_grad_kernel;
public:
    GD();
    void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) override;
};

class Adam : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<Tensor*> m; // first moment
    std::vector<Tensor*> v; // second moment
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, float, float, float, int, float, int> adam_update_kernel;
    cl::make_kernel<cl::Buffer, int> zero_grad_kernel;

public:
    Adam(float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) override;
    void initialize_momentum(std::vector<Tensor*> parameters) override;
};
