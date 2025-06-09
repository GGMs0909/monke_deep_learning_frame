#include "model.h"

Model::Model(Loss& lossfunction, Optimizer& optimizer) :
	gradient_norm_kernel(opencl_runtime::getInstance().get_program(), "calculate_gradient_norm")
{
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
	std::cout << "Initializing inputs." << std::endl;
	inputs = std::vector<Tensor>(layers.size() + 1); // +1 for the output layer
	grad_inputs = std::vector<Tensor>(layers.size() + 1); // +1 for the output layer
	grad_inputs[0] = Tensor({ 1, 1, 1 }); // Initialize grad_inputs[0] with a dummy shape
	std::cout << "Initializing parameters." << std::endl;
	parameters = std::vector<Tensor*>(); // Store model parameters
	grad_parameters = std::vector<Tensor*>(); // Store gradients of model parameters
	// Initialize inputs and grad_inputs for each layer
	std::cout << "Compiling " << layers.size() << " layers." << std::endl;
	for (int i = 0; i < layers.size(); ++i) {
		if (!layers[i]) {
			std::cerr << "Error: layer[" << i << "] is nullptr!" << std::endl;
			continue; // Á×§K crash
		}
		std::cout << i << " " << layers[i]->get_name() << std::endl;
		layers[i]->Get_Tensor(inputs[i + 1]);
		layers[i]->Get_Tensor(grad_inputs[i + 1]); // Get the output tensor shape

		layers[i]->get_parameters(parameters, grad_parameters); // Get parameters and gradients from the layer

	}
	std::cout << "Model layers compiled." << std::endl;
	// Initialize optimizer moments if needed
	std::cout << "Initializing optimizer moments." << std::endl;
	optimizer->initialize_moments(parameters);
	reset();
	opencl_runtime::getInstance().get_queue().finish();
}

void Model::forward(const Tensor& input, Tensor& output) {
	// Forward pass through the model
	inputs[0] = input; // Set the input for the first layer
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->forward(inputs[i], inputs[i + 1]);
	}
	output = inputs.back(); // The output of the last layer
	output.transfer_to_cpu(); // Ensure output is on CPU for loss calculation
	opencl_runtime::getInstance().get_queue().finish();
}
float Model::forward_with_loss(const Tensor& input, Tensor& output, const Tensor& real) {
	// Forward pass with loss calculation
	std::cout << "Forward pass with loss calculation." << std::endl;
	inputs[0] = input; // Set the input for the first layer

	for (size_t i = 0; i < layers.size(); ++i) {
		//std::cout << layers[i]->get_name() << std::endl;
		layers[i]->forward(inputs[i], inputs[i + 1]);
	}

	output = inputs.back(); // The output of the last layer
	output.transfer_to_cpu(); // Ensure output is on CPU for loss calculation
	opencl_runtime::getInstance().get_queue().finish();
	return lossfunction->calculate(output, real);
	
}
void Model::backward(const Tensor& prep, const Tensor& real) {
	// Backward pass through the model
	lossfunction->backward(prep, real, grad_inputs.back());
	for (int i = layers.size() - 1; i > 0; --i) {
		// Backward pass through each layer
		//std::cout << layers[i]->get_name() << std::endl;
		layers[i]->backward(grad_inputs[i + 1], inputs[i], grad_inputs[i]);
	}
	opencl_runtime::getInstance().get_queue().finish();
}
void Model::update() {
	
	optimizer->update(parameters, grad_parameters);
	opencl_runtime::getInstance().get_queue().finish();
	// Reset gradients in update
}

float Model::get_gradient_norm() {
	// Calculate the norm of the gradients
	float norm = 0;
	for (int i = 0; i < grad_parameters.size(); ++i) {
		grad_parameters[i]->transfer_to_cpu(); // Ensure gradients are on CPU for norm calculation
		for (float& val : grad_parameters[i]->data) {
			norm += val; // Reset gradients to zero before calculating norm
		}

	}
	return norm;
}

void Model::reset() {
	optimizer->reset(parameters, grad_parameters);
	opencl_runtime::getInstance().get_queue().finish();

}

void Model::set_parameters(const std::vector<float> data) {
	// Set model parameters from a vector of floats
	int offset = 0;
	if (data.size() != parameters.size()) {
		throw std::invalid_argument("Data size does not match number of parameters");
	}
	for (size_t i = 0; i < parameters.size(); ++i) {
		for (int j = 0; j < parameters[i]->data.size(); ++j) {
			parameters[i]->data[i] = data[offset + i]; // Set parameter values
		}
		offset += parameters[i]->data.size(); // Update offset for next parameter
		parameters[i]->transfer_to_gpu(); // Transfer updated parameters to GPU
	}
}


//for debugging purposes
void Model::print_parameters() const {
	std::cout << "Model Parameters:" << std::endl;
	for (const auto& param : parameters) {
		param->print(10); // Print first 10 elements
	}
}
void Model::print_grad_parameters() const {
	std::cout << "Model Gradient Parameters:" << std::endl;
	for (const auto& grad_param : grad_parameters) {
		grad_param->print(10); // Print first 10 elements
	}
}
void Model::print_grad_inputs()  {
	std::cout << "Model Gradient Inputs:" << std::endl;
	for (auto& grad_input : grad_inputs) {
		grad_input.print(10); // Print first 10 elements
	}
}