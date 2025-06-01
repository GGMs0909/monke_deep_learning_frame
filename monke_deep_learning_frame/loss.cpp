//loss.cpp
#include "loss.h"
#include <cmath>
#include <stdexcept>

Loss::Loss() {
	// Constructor implementation
}
Loss::~Loss() {
	// Destructor implementation
}

MeanSquaredError::MeanSquaredError(int input_size) : input_size(input_size) {
	// Constructor implementation
}
MeanSquaredError::~MeanSquaredError() {
	// Destructor implementation
}
float MeanSquaredError::calculate(const Tensor& pred, const Tensor& real) {
	if (pred.size() != real.size()) {
		throw std::invalid_argument("Predicted and real tensors must have the same size.");
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
		throw std::invalid_argument("CrossEntropyLoss::calculate: 預測張量和真實張量的大小必須相同。");
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
		throw std::invalid_argument("CrossEntropyLoss::backward: 預測張量、真實張量和梯度輸出張量的大小必須相同。");
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