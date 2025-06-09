#include "layer.h"
#include <random>


static float random_normal_float(float mean, float stddev) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<float> dist(mean, stddev);
	return dist(gen);
}

Layer::Layer() {
	// Constructor implementation
}
Layer::~Layer() {
	// Destructor implementation
}

Relu::Relu(int input_size)
    : input_size(input_size), 
    Relu_forward_kernel(opencl_runtime::getInstance().get_program(), "relu_forward"),
    Relu_backward_kernel(opencl_runtime::getInstance().get_program(), "relu_backward"),
    enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size)),
    enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size))
{

}
Relu::~Relu() {

}
std::string Relu::get_name() {
	return "relu " + std::to_string(input_size) +" -> " + std::to_string(input_size);
}
void Relu::Get_Tensor(Tensor& output) {
	output = Tensor({ input_size });
}
void Relu::forward(const Tensor& input, Tensor& output) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Relu_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), input_size);
}
void Relu::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Relu_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), input.get_buffer(), grad_input.get_buffer(), input_size);
}
void Relu::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {

}


Softmax::Softmax(int input_size)
	: input_size(input_size), 

	Softmax_forward_kernel(opencl_runtime::getInstance().get_program(), "softmax_forward"),
	Softmax_backward_kernel(opencl_runtime::getInstance().get_program(), "softmax_backward"),

	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size)),
	enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size))
{

}
Softmax::~Softmax() {

}
std::string Softmax::get_name() {
	return "softmax " + std::to_string(input_size) + " -> " + std::to_string(input_size);
}
void Softmax::Get_Tensor(Tensor& output) {
	output = Tensor({ input_size });
}
void Softmax::forward(const Tensor& input, Tensor& output) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Softmax_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), input_size);

	output_backup = output.get_buffer();
}
void Softmax::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Softmax_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), output_backup, grad_input.get_buffer(), input_size);
}
void Softmax::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {
}

Dense::Dense(int input_size, int output_size)
	: input_size(input_size), output_size(output_size),

	Dense_forward_kernel(opencl_runtime::getInstance().get_program(), "dense_forward"),
	Dense_backward_wb_kernel(opencl_runtime::getInstance().get_program(), "dense_backward_wb"),
	Dense_backward_input_kernel(opencl_runtime::getInstance().get_program(), "dense_backward_input"),
	update_vector_kernel(opencl_runtime::getInstance().get_program(), "update_vector"),

	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_size)),
	enqueue_args_backward_wb(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_size)),
	enqueue_args_backward_input(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size)),
	enqueue_args_update_weights(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_size* input_size)),
	enqueue_args_update_biases(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_size))
{
    weights = Tensor({ output_size, input_size });
    biases = Tensor({ output_size });
    grad_weights = Tensor({ output_size, input_size });
    grad_biases = Tensor({ output_size });
	grad_weights.transfer_to_gpu();
	grad_biases.transfer_to_gpu();
	grad_weights.get_buffer();
	grad_biases.get_buffer();


    for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				weights.get({ i, j }) = random_normal_float(0.0f, sqrt(2.0f / input_size));
				
			}
    }
    for (int j = 0; j < output_size; ++j) {
			biases.get({ j }) = random_normal_float(0.0f, sqrt(2.0f / output_size));
			
    }
	weights.transfer_to_gpu();
	biases.transfer_to_gpu();
}
Dense::~Dense() {

}
std::string Dense::get_name() {
	return "dense " + std::to_string(input_size) + " -> " + std::to_string(output_size);
}
void Dense::Get_Tensor(Tensor& output) {
	output = Tensor({ output_size });
}
void Dense::forward(const Tensor& input, Tensor& output) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Dense_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), weights.get_buffer(), biases.get_buffer(), input_size, output_size);
}
void Dense::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Dense_backward_wb_kernel(enqueue_args_backward_wb, grad_output.get_buffer(), input.get_buffer(), grad_weights.get_buffer(), grad_biases.get_buffer(), input_size, output_size);
	Dense_backward_input_kernel(enqueue_args_backward_input, grad_output.get_buffer(), weights.get_buffer(), grad_input.get_buffer(), input_size, output_size);
}
void Dense::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {

	weights.print();
	biases.print();
	grad_weights.print();
	grad_biases.print();
	parameters.push_back(&weights);
	parameters.push_back(&biases);
	grad_parameters.push_back(&grad_weights);
	grad_parameters.push_back(&grad_biases);
}


Convolution::Convolution(int input_channels, int input_size, int output_channels, int kernel_size)
	: input_channels(input_channels), input_size(input_size), output_channels(output_channels), kernel_size(kernel_size),

	Convolution_forward_kernel(opencl_runtime::getInstance().get_program(), "convolution_forward"),
	Convolution_backward_weights_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_weights"),
	Convolution_backward_biases_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_biases"),
	Convolution_backward_input_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_input"),
	update_vector_kernel(opencl_runtime::getInstance().get_program(), "update_vector"),

	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels, input_size - kernel_size + 1, input_size - kernel_size + 1)),
	enqueue_args_backward_weights(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels, input_channels, kernel_size*kernel_size)),
	enqueue_args_backward_biases(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels)),
	enqueue_args_backward_input(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, input_size, input_size)),
	enqueue_args_update_weights(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels* input_channels* kernel_size* kernel_size)),
	enqueue_args_update_biases(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels))
{
	weights = Tensor({ output_channels, input_channels, kernel_size, kernel_size });
	biases = Tensor({ output_channels });
	grad_weights = Tensor({ output_channels, input_channels, kernel_size, kernel_size });
	grad_biases = Tensor({ output_channels });
	for (int i = 0; i < output_channels; ++i) {
		for (int j = 0; j < input_channels; ++j) {
			for (int k = 0; k < kernel_size; ++k) {
				for (int l = 0; l < kernel_size; ++l) {
					weights.get({ i,j,k,l }) = random_normal_float(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));

				}
			}
		}
		biases.get({ i }) = random_normal_float(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));
		
	}
	weights.transfer_to_gpu();
	biases.transfer_to_gpu();
	grad_weights.transfer_to_gpu();
	grad_biases.transfer_to_gpu();
}
Convolution::~Convolution() {

}
std::string Convolution::get_name() {
	return "convolution " + std::to_string(input_channels) + "x" + std::to_string(input_size) + "x" + std::to_string(input_size) + 
		" -> " + std::to_string(output_channels) + "x" + std::to_string(input_size-kernel_size+1)+"x" + std::to_string(input_size - kernel_size + 1);
}
void Convolution::Get_Tensor(Tensor& output) {
	output = Tensor({ output_channels, input_size - kernel_size + 1, input_size - kernel_size + 1 });
}
void Convolution::forward(const Tensor& input, Tensor& output) {
	opencl_runtime::getInstance().get_queue().finish();

	Convolution_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), weights.get_buffer(), biases.get_buffer(), input_channels, input_size, output_channels, kernel_size);
}
void Convolution::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Convolution_backward_weights_kernel(enqueue_args_backward_weights, grad_output.get_buffer(), input.get_buffer(), grad_weights.get_buffer(), input_channels, input_size, output_channels, kernel_size);
	Convolution_backward_biases_kernel(enqueue_args_backward_biases, grad_output.get_buffer(), grad_biases.get_buffer(), output_channels, input_size - kernel_size + 1);
	Convolution_backward_input_kernel(enqueue_args_backward_input, grad_output.get_buffer(), weights.get_buffer(), grad_input.get_buffer(), input_channels, input_size, output_channels, kernel_size);
}
void Convolution::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {
	weights.transfer_to_cpu();
	biases.transfer_to_cpu();
	grad_weights.transfer_to_cpu();
	grad_biases.transfer_to_cpu();
	weights.print();
	biases.print();
	grad_weights.print();
	grad_biases.print();
	parameters.push_back(&weights);
	parameters.push_back(&biases);
	grad_parameters.push_back(&grad_weights);
	grad_parameters.push_back(&grad_biases);
}


Pooling::Pooling(int input_channels, int input_size, int pool_size)
	: input_channels(input_channels), input_size(input_size), pool_size(pool_size),

	Pooling_forward_kernel(opencl_runtime::getInstance().get_program(), "pooling_forward"),
	Pooling_backward_kernel(opencl_runtime::getInstance().get_program(), "pooling_backward"),
	gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "gradient_reset"),

	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, (input_size / pool_size), (input_size / pool_size))),
	enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, (input_size / pool_size), (input_size / pool_size))),
	enqueue_args_reset_gradients(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels* input_size* input_size))
{
	int output_dim = input_size / pool_size;
	if (output_dim <= 0) {
		throw std::invalid_argument("Invalid input_size or pool_size for pooling layer.");
	}
	max_indices = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, sizeof(int) * input_channels * output_dim * output_dim * 2);

}
Pooling::~Pooling() {

}
std::string Pooling::get_name() {
	return "pooling " + std::to_string(input_channels) + "x" + std::to_string(input_size) + "x" + std::to_string(input_size) +
		" -> " + std::to_string(input_channels) + "x" + std::to_string(input_size / pool_size) + "x" + std::to_string(input_size / pool_size);
}
void Pooling::Get_Tensor(Tensor& output) {
	int output_dim = input_size / pool_size;
	output = Tensor({ input_channels, output_dim, output_dim });
}
void Pooling::forward(const Tensor& input, Tensor& output) {

	opencl_runtime::getInstance().get_queue().finish(); 
	Pooling_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), max_indices, input_channels, input_size, pool_size);
}
void Pooling::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	opencl_runtime::getInstance().get_queue().finish(); 

	gradient_reset_kernel(enqueue_args_reset_gradients, grad_input.get_buffer(), input_channels * input_size * input_size);
	Pooling_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), grad_input.get_buffer(), max_indices, input_channels, input_size, pool_size);
	
	
}
void Pooling::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {

}


Flatten_3D::Flatten_3D(int input_channels, int input_size)
	: input_channels(input_channels), input_size(input_size)
{ 
	
}
Flatten_3D::~Flatten_3D() {

}
std::string Flatten_3D::get_name() {
	return "flatten3D " + std::to_string(input_channels) + "x" + std::to_string(input_size) + "x" + std::to_string(input_size) +
		" -> " + std::to_string(input_channels*input_size*input_size);
}
void Flatten_3D::Get_Tensor(Tensor& output) {
	int total_elements = input_channels * input_size * input_size;
	output = Tensor({ total_elements });
}
void Flatten_3D::forward(const Tensor& input, Tensor& output) {


	output.share_buffer_and_reshape(input, { input_channels * input_size * input_size });
}
void Flatten_3D::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {

	grad_input.share_buffer_and_reshape(grad_output, { input_channels, input_size, input_size });
}
void Flatten_3D::get_parameters(std::vector<Tensor*>& parameters, std::vector<Tensor*>& grad_parameters) {

}

	
