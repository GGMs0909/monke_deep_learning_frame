#include "optimizer.h"
#include <stdexcept>

GradientDescent::GradientDescent(float learning_rate) : learning_rate(learning_rate), kernel(opencl_runtime::getInstance().get_program(), "gradient_decent") {
	// Initialize
}
GradientDescent::~GradientDescent() {
	// Clean up resources if needed
}
void GradientDescent::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
	// Update weights using OpenCL kernel
	for (int i = 0; i < parameters.size(); ++i) {
		if (parameters[i]->size() != grad_parameters[i]->size()) {
			throw std::invalid_argument("Parameter and gradient sizes do not match.");
		}
		kernel(cl::EnqueueArgs(opencl_runtime::getInstance().get_queue(), cl::NDRange(parameters[i]->size())),
			parameters[i]->get_buffer(), grad_parameters[i]->get_buffer(), learning_rate, parameters[i]->size());
	}
	// Reset gradients done in kernel
}