//loss.cpp
#include "loss.h"
#include <cmath>
#include <stdexcept>
#include "model.h"

Loss::Loss() {
	// Constructor implementation
}
Loss::~Loss() {
	// Destructor implementation
}
LossWithNormalization::LossWithNormalization(Model* model, Loss* loss_function, float normalization_factor)
	: model_(model), loss_function_(loss_function), normalization_factor_(normalization_factor), 
	normalization_backward_kernel(opencl_runtime::getInstance().get_program(), "normalization_backward")
{
	// Constructor implementation
	if (!model_) {
		throw std::invalid_argument("Model pointer cannot be null.");
	}
	if (!loss_function_) {
		throw std::invalid_argument("Loss function pointer cannot be null.");
	}
	// Initialize parameters and gradients for normalization
	parameters = model_->get_parameters(); // Get model parameters for normalization
	grad_parameters = model_->get_grad_parameters(); // Get gradients of model parameters for normalization
}
LossWithNormalization::~LossWithNormalization() {
	// Destructor implementation
}
float LossWithNormalization::calculate(const Tensor& pred, const Tensor& real) {
	float loss = loss_function_->calculate(pred, real);
	
	return loss;
}
void LossWithNormalization::backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) {
	if (pred.size() != real.size() || pred.size() != grad_output.size()) {
		throw std::invalid_argument("Predicted, real, and gradient output tensors must have the same size.");
	}
	loss_function_->backward(pred, real, grad_output);

	// Normalize gradients
	opencl_runtime::getInstance().get_queue().finish();
	for (int i = 0; i < parameters.size(); ++i) {
		if (parameters[i]->size() != grad_parameters[i]->size()) {
			throw std::invalid_argument("Parameter and gradient sizes do not match.");
		}
		normalization_backward_kernel(cl::EnqueueArgs(opencl_runtime::getInstance().get_queue(), cl::NDRange(grad_parameters[i]->size())),
			grad_parameters[i]->get_buffer(), parameters[i]->get_buffer(), grad_parameters[i]->size(), normalization_factor_);
	}

	grad_output.transfer_to_gpu();
}
std::string LossWithNormalization::get_name() const {
	return "LossWithNormalization(" + loss_function_->get_name() + ")";
}


MeanSquaredError::MeanSquaredError(int input_size) : input_size(input_size) {
	// Constructor implementation
}
MeanSquaredError::~MeanSquaredError() {
	// Destructor implementation
}
float MeanSquaredError::calculate(const Tensor& pred, const Tensor& real) {
	if (pred.size() != real.size()) {
		throw std::invalid_argument("Predicted and real tensors must have the same size.(" +std::to_string(pred.size()) + "vs" + std::to_string(real.size())+")");
	}
	float loss = 0.0f;
	for (int i = 0; i < input_size; ++i) {
		float diff = pred.get({ i }) - real.get({ i });
		loss += diff * diff;
	}
	return loss / input_size;
}
void MeanSquaredError::backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) {
	if (pred.size() != real.size() || pred.size() != grad_output.size()) {
		throw std::invalid_argument("Predicted, real, and gradient output tensors must have the same size.");
	}
	for (int i = 0; i < input_size; ++i) {
		grad_output.get({ i }) = 2.0f * (pred.get({ i }) - real.get({ i })) / input_size;
	}
	grad_output.transfer_to_gpu();
}
std::string MeanSquaredError::get_name() const {
	return "MeanSquaredError";
}
CrossEntropyLoss::CrossEntropyLoss(int input_size) : input_size(input_size) {
	// Constructor implementation
}
CrossEntropyLoss::~CrossEntropyLoss() {
	// Destructor implementation
}
float CrossEntropyLoss::calculate(const Tensor& pred, const Tensor& real) {
	if (pred.size() != real.size()) {
		throw std::invalid_argument("Predicted and real tensors must have the same size.(" + std::to_string(pred.size()) + "vs" + std::to_string(real.size()) + ")");
	}
	float loss = 0.0f;
	const float epsilon = 1e-9f; 
	for (int i = 0; i < pred.size(); ++i) {
		float clamped_pred = std::fmax(pred.get({ i }), epsilon);
		clamped_pred = std::fmin(clamped_pred, 1.0f - epsilon); 

		loss -= real.get({ i }) * std::log(clamped_pred);
	}
	return loss;
}
void CrossEntropyLoss::backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) {
	if (pred.size() != real.size() || pred.size() != grad_output.size()) {
		throw std::invalid_argument("CrossEntropyLoss::backward");
	}
	const float epsilon = 1e-9f;
	for (int i = 0; i < pred.size(); ++i) {
		float clamped_pred = std::fmax(pred.get({ i }), epsilon);
		grad_output.get({ i }) = -real.get({ i }) / clamped_pred;
	}

	grad_output.transfer_to_gpu();
}

std::string CrossEntropyLoss::get_name() const {
	return "CrossEntropyLoss";
}