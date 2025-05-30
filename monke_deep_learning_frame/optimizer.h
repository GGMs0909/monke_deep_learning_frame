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
};

class GradientDescent : public Optimizer {
public:
	GradientDescent(float learning_rate);
	~GradientDescent() override;
	void update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) override;
private:
	float learning_rate; // Learning rate for the optimizer
	cl::make_kernel<cl::Buffer, cl::Buffer, float, int> kernel; // OpenCL kernel for gradient descent
};
