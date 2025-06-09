#pragma once
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "Tensor.h"
#include <iostream>
#include <vector>

class Model {
public:
	Model();
	
	~Model();
	bool compiled = false;
	void add_layer(Layer* layer);
	void compile(std::vector<int> input_shape, Loss& lossfunction, Optimizer& optimizer);
	void forward(const Tensor& input, Tensor& output);
	float forward_with_loss(const Tensor& input, Tensor& output, const Tensor& real); // Forward pass with loss calculation
	void backward(const Tensor& prep, const Tensor& real); // Backward pass through the model
	void update(); // Update model parameters using the optimizer
	void reset();
	float get_gradient_norm(); // Calculate the norm of the gradients
	void set_parameters(const std::vector<float> data);
	void extract_parameters(std::vector<float>& data) const; // Extract model parameters into a vector
	//for debugging
	void print_parameters() const; // Print model parameters for debugging
	void print_grad_parameters() const; // Print gradients of model parameters for debugging
	void print_grad_inputs() ; // Print gradients for each layer for debugging

private:
	std::vector<Layer*> layers; // Store layers in a vector
	std::vector<Tensor> inputs; // Store inputs for each layer
	std::vector<Tensor> grad_inputs; // Store gradients for each layer
	Loss* lossfunction; // Pointer to the loss function
	std::vector<Tensor*> parameters; // Store model parameters (weights and biases)
	std::vector<Tensor*> grad_parameters; // Store gradients of model parameters
	Optimizer* optimizer; // Pointer to the optimizer
	cl::make_kernel<cl::Buffer, cl::Buffer, int> gradient_norm_kernel; // OpenCL kernel for calculating gradient norm
};

