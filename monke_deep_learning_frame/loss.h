//loss.h
#pragma once
#include "opencl_runtime.h"
#include "Tensor.h"
#include <vector>

class Model;// forward declaration

class Loss {
public:
	Loss();
	virtual ~Loss();
	// Calculate the loss value
	virtual float calculate(const Tensor& pred, const Tensor& real) = 0;
	// Calculate the gradient of the loss with respect to the output
	virtual void backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) = 0;
	// Get the name of the loss function
	virtual std::string get_name() const = 0;
};
class LossWithNormalization : public Loss {
public:
	LossWithNormalization(Model* model, Loss* loss_function, float normalization_factor);
	~LossWithNormalization() override;
	float calculate(const Tensor& pred, const Tensor& real) override;
	void backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) override;
	std::string get_name() const override;
private:

	float normalization_factor_;
	Model* model_; // Pointer to the model for normalization factor calculation
	Loss* loss_function_; // Pointer to the actual loss function used for calculation
	cl::make_kernel < cl::Buffer, cl::Buffer, int, float > normalization_backward_kernel; // OpenCL kernel for normalization
};

class MeanSquaredError : public Loss {
public:
	MeanSquaredError(int input_size);
	~MeanSquaredError() override;
	float calculate(const Tensor& pred, const Tensor& real) override;
	void backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) override;
	std::string get_name() const override;
private:
	int input_size;
};

class CrossEntropyLoss : public Loss {
public:
	CrossEntropyLoss(int input_size);
	~CrossEntropyLoss() override;
	float calculate(const Tensor& pred, const Tensor& real) override;
	void backward(const Tensor& pred, const Tensor& real, Tensor& grad_output) override;
	std::string get_name() const override;
private:
	int input_size;

};

