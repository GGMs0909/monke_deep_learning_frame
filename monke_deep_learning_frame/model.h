#pragma once
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "Tensor.h"
#include <iostream>
#include <vector>

class Model {
public:
	Model(Loss& lossfunction, Optimizer& optimizer);
	~Model();
	void add_layer(Layer* layer);
	void compile();
	void forward(const Tensor& input, Tensor& output);
	void forward(const Tensor& input, Tensor& output, const Tensor& real, float& loss_value); // Forward pass with loss calculation
	void backward(const Tensor& prep, const Tensor& real); // Backward pass through the model
	void update(); // Update model parameters using the optimizer

private:
	std::vector<Layer*> layers; // Store layers in a vector
	std::vector<Tensor> inputs; // Store inputs for each layer
	std::vector<Tensor> grad_inputs; // Store gradients for each layer
	Loss* lossfunction; // Pointer to the loss function
	std::vector<Tensor*> parameters; // Store model parameters (weights and biases)
	std::vector<Tensor*> grad_parameters; // Store gradients of model parameters
	Optimizer* optimizer; // Pointer to the optimizer
};

