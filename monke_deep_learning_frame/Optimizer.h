//optimizer.h
#pragma once
#include "opencl_runtime.h"
//optimizer.h
#include "Tensor.h"
#include "layer.h" 
#include <vector>

class Optimizer {
public:
	Optimizer() = default;
	virtual ~Optimizer() = default;
	// Apply the optimizer to the model's parameters
	virtual void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) = 0;
	// Reset gradients will done in update
    virtual void initialize_moments(const std::vector<Tensor*>& parameters) {}
    virtual void reset(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) = 0;
};

class GradientDescent : public Optimizer {
public:
	GradientDescent(float learning_rate);
	~GradientDescent() override;
	void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) override;
    void reset(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) override;
private:
	float learning_rate; // Learning rate for the optimizer
	cl::make_kernel<cl::Buffer, cl::Buffer, float, int> kernel; // OpenCL kernel for gradient descent
	cl::make_kernel < cl::Buffer, int> gradient_reset_kernel; // OpenCL kernel for resetting gradients 
};

class Adam : public Optimizer {
public:

    Adam(float learning_rate, float beta1, float beta2, float epsilon);
    ~Adam() override;


    void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) override;


    void initialize_moments(const std::vector<Tensor*>& parameters) override;
    void reset(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) override;

private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t; 

    std::vector<Tensor> m; // First moment vector
    std::vector<Tensor> v; // Second moment vector

    // OpenCL Kernel for Adam update
    // Args: param_buffer, grad_buffer, m_buffer, v_buffer, learning_rate, beta1, beta2, epsilon, t, size
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, float, float, float, float, int, int> adam_kernel;
    cl::make_kernel < cl::Buffer, int> gradient_reset_kernel; // OpenCL kernel for resetting gradients 
};
