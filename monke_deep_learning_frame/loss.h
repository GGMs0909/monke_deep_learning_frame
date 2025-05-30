//loss.h
#pragma once
#include "opencl_runtime.h"
#include "Tensor.h"
#include <vector>

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

