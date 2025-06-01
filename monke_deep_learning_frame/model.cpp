#include "model.h"

Model::Model(Loss& lossfunction, Optimizer& optimizer) {
	// Initialize the model
	this->lossfunction = &lossfunction;
	this->optimizer = &optimizer;
}
Model::~Model() {
	// Clean up resources
	for (Layer* layer : layers) {
		delete layer;
	}
}
void Model::add_layer(Layer* layer) {
	// Add a layer to the model
	std::cout << "Adding layer: " << layer->get_name() << std::endl;
	layers.push_back(layer);
}
void Model::compile() {
	// Compile the model, prepare inputs and parameters
	inputs.clear();
	grad_inputs.clear();
	parameters.clear();
	grad_parameters.clear();
	inputs = std::vector<Tensor>(layers.size() + 1); // +1 for the output layer
	grad_inputs = std::vector<Tensor>(layers.size() + 1); // +1 for the output layer
	// Initialize inputs and grad_inputs for each layer
	for (int i = 0; i < layers.size(); ++i) {
		layers[i]->Get_Tensor(inputs[i + 1]);
		layers[i]->Get_Tensor(grad_inputs[i + 1]); // Get the output tensor shape
		if (inputs[i + 1].cl_buffer()) {
			std::cout << "Layer " << i << " input tensor initialized with shape: ";
			inputs[i + 1].print(10); // Print the shape of the input tensor
		}
		else {
			std::cout << "Layer " << i << " input tensor not initialized." << std::endl;
		}
		if (grad_inputs[i + 1].cl_buffer()) {
			std::cout << "Layer " << i << " grad input tensor initialized with shape: ";
			grad_inputs[i + 1].print(10); // Print the shape of the grad input tensor
		}
		else {
			std::cout << "Layer " << i << " grad input tensor not initialized." << std::endl;
		}
		if (layers[i]->has_parameter) {
			parameters.push_back(&layers[i]->weights);
			parameters.push_back(&layers[i]->biases);
			grad_parameters.push_back(&layers[i]->grad_weights);
			grad_parameters.push_back(&layers[i]->grad_biases);
		}
	}

}
void Model::forward(const Tensor& input, Tensor& output) {
	// Forward pass through the model
	inputs[0] = input; // Set the input for the first layer
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->forward(inputs[i], inputs[i + 1]);
	}
	output = inputs.back(); // The output of the last layer
}
void Model::forward(const Tensor& input, Tensor& output, const Tensor& real, float& loss_value) {
	// Forward pass with loss calculation
	std::cout << "Forward pass with loss calculation." << std::endl;
	inputs[0] = input; // Set the input for the first layer

	for (size_t i = 0; i < layers.size(); ++i) {
		std::cout << layers[i]->get_name() << std::endl;
		layers[i]->forward(inputs[i], inputs[i + 1]);
	}

	output = inputs.back(); // The output of the last layer
	loss_value = lossfunction->calculate(output, real);
}
void Model::backward(const Tensor& prep, const Tensor& real) {
	// Backward pass through the model
	lossfunction->backward(prep, real, grad_inputs.back());
	for (int i = layers.size() - 1; i > 0; --i) {
		// Backward pass through each layer
		std::cout << layers[i]->get_name() << std::endl;
		layers[i]->backward(grad_inputs[i + 1], inputs[i], grad_inputs[i]);
	}
}
void Model::update() {
	// Update model parameters using the optimizer
	optimizer->update(parameters, grad_parameters);
	// Reset gradients in update
}