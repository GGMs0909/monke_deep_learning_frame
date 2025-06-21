#pragma once
#include "opencl_runtime.h"
#include "Tensor.h"
#include <vector>
#include <string>
#include <stdexcept>
//Relu(int input_size);
//Softmax(int input_size);
//Dense(int input_size, int output_size);
//Convolution(int input_channels, int input_size, int output_channels, int kernel_size);
//Pooling(int input_channels, int input_size, int pool_size);
//Flatten_3D(int input_channels, int input_size);


class Layer {

public:
	Layer();
	virtual ~Layer();
	virtual std::string get_name() = 0;
	virtual void Get_Tensor(Tensor& output) = 0;
	virtual void forward(const Tensor& input, Tensor& output) = 0;
	virtual void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) = 0;
	virtual void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) = 0;
	virtual void set_training(bool is_training);

protected:
	bool is_training_ = true;

};

class Dropout : public Layer {
public:
	Dropout(float dropout_rate, int input_size);
	~Dropout() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;
private:
	float  dropout_rate_;
	int input_size_;
	Tensor mask; // Mask to store the dropout mask
	//__kernel void scale_forward(__global float* input, __global float* output, int size, float scale)
	cl::make_kernel <cl::Buffer, cl::Buffer, int, float> scale_kernel;
	//__kernel void dropout_forward(__global float* input, __global float* output, __global float* mask, int size, float dropout_rate, int seed)
	cl::make_kernel < cl::Buffer, cl::Buffer, cl::Buffer, int, float, int> Dropout_forward_kernel;
	//__kernel void dropout_backward(__global float* grad_output, __global float* grad_input, __global float* mask, int size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, float> Dropout_backward_kernel;
};

class Scale : public Layer {
public:
	Scale(int input_size, float scale_size);
	~Scale() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;
private:
	int input_size_;
	float scale_size_;
	cl::make_kernel<cl::Buffer, cl::Buffer, int, float> Scale_forward_kernel;
	cl::make_kernel<cl::Buffer, cl::Buffer, int, float> Scale_backward_kernel;
};

class Relu : public Layer { // deleted Relu_1D and Relu_3D, now using Relu
public:
	Relu(int input_size);
	~Relu() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;


private:
	int input_size;
	cl::make_kernel<cl::Buffer, cl::Buffer, int> Relu_forward_kernel;
	// Relu_forward_kernel(input, output, size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> Relu_backward_kernel;
	// Relu_backward_kernel(grad_output, input, grad_input, size)
	cl::EnqueueArgs enqueue_args_forward;
	cl::EnqueueArgs enqueue_args_backward;
};
class Softmax : public Layer {
public:
	Softmax(int input_size);
	~Softmax() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;



private:
	int input_size;
	cl::make_kernel<cl::Buffer, cl::Buffer, int> Softmax_forward_kernel;
	// Softmax_forward_kernel(input, output, size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> Softmax_backward_kernel;
	// Softmax_backward_kernel(grad_output, input, grad_input, size)
	cl::EnqueueArgs enqueue_args_forward;
	cl::EnqueueArgs enqueue_args_backward;
	cl::Buffer output_backup; // Store the output of the forward pass for use in the backward pass
};
class Dense : public Layer {
public:
	Dense(int input_size, int output_size);
	~Dense() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;

	Tensor weights; // weights are stored in a Tensor
	Tensor biases; // biases are stored in a Tensor
	Tensor grad_weights; // grad_weights are stored in a Tensor
	Tensor grad_biases; // grad_biases are stored in a Tensor
private:
	int input_size;
	int output_size;
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> Dense_forward_kernel;
	// Dense_forward_kernel(input, output, weights, biases, input_size, output_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> Dense_backward_wb_kernel;
	// Dense_backward_kernel(grad_output, input, grad_weights, grad_biases, input_size, output_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> Dense_backward_input_kernel;
	// Dense_backward_input_kernel(grad_putput, weights, grad_input, input_size, output_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, float, int> update_vector_kernel;
	// update_vector_kernel(weights, grad_weights, learning_rate, input_size * output_size);
	cl::EnqueueArgs enqueue_args_forward;
	cl::EnqueueArgs enqueue_args_backward_wb;
	cl::EnqueueArgs enqueue_args_backward_input;
	cl::EnqueueArgs enqueue_args_update_weights;
	cl::EnqueueArgs enqueue_args_update_biases;


};

class Convolution : public Layer {
public:
	Convolution(int input_channels, int input_size, int output_channels, int kernel_size);
	~Convolution() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;

	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;

	Tensor weights; // weights are stored in a Tensor
	Tensor biases; // biases are stored in a Tensor
	Tensor grad_weights; // grad_weights are stored in a Tensor
	Tensor grad_biases; // grad_biases are stored in a Tensor
private:
	int input_channels;
	int input_size;
	int output_channels;
	int kernel_size;
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> Convolution_forward_kernel;
	// Convolution_forward_kernel(input, output, weights, biases, input_channels, input_size, output_channels, kernel_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> Convolution_backward_weights_kernel;
	// Convolution_backward_kernel(grad_output, input, grad_weights, input_channels, input_size, output_channels, kernel_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, int, int> Convolution_backward_biases_kernel;
	// Convolution_backward_biases_kernel(grad_output, grad_biases, output_channels, output_size = input_size-kernel_size+1)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> Convolution_backward_input_kernel;
	// Convolution_backward_input_kernel(grad_output, weights, grad_input, input_channels, input_size, output_channels, kernel_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, float, int> update_vector_kernel;
	// update_vector_kernel(grad_weights, weights, learning_rate, input_channels * kernel_size * kernel_size);
	// update_vector_kernel(grad_biases, biases, learning_rate, output_channels);

	cl::EnqueueArgs enqueue_args_forward;
	cl::EnqueueArgs enqueue_args_backward_weights;
	cl::EnqueueArgs enqueue_args_backward_biases;
	cl::EnqueueArgs enqueue_args_backward_input;
	cl::EnqueueArgs enqueue_args_update_weights;
	cl::EnqueueArgs enqueue_args_update_biases;


};

class Pooling : public Layer {
public:
	Pooling(int input_channels, int input_size, int pool_size);
	~Pooling() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;

private:
	int input_channels;
	int input_size;
	int pool_size;
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int> Pooling_forward_kernel;
	// Pooling_forward_kernel(input, output, max_indices, input_channels, input_size, pool_size)
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int> Pooling_backward_kernel;
	// Pooling_backward_kernel(grad_output, grad_input, max_indices, input_channels, input_size, pool_size)
	cl::make_kernel<cl::Buffer, int> gradient_reset_kernel;
	// gradient_reset_kernel(grad_input,input_channels*input_size*input_size)
	cl::EnqueueArgs enqueue_args_forward;
	cl::EnqueueArgs enqueue_args_backward;
	cl::EnqueueArgs enqueue_args_reset_gradients;
	cl::Buffer max_indices; // Store the relative (i,j) index of the maximum value in the pool_size x pool_size window for each output element
};

class Flatten_3D : public Layer {
public:
	Flatten_3D(int input_channels, int input_size);
	~Flatten_3D() override;
	std::string get_name() override;
	void Get_Tensor(Tensor& output) override;
	void forward(const Tensor& input, Tensor& output) override;
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override;
	void get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) override;

private:
	int input_channels;
	int input_size;
};






