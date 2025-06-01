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
    : input_size(input_size), // ��l�� total_elements_
    // �b�o�̪�����l�� cl::make_kernel ����
    Relu_forward_kernel(opencl_runtime::getInstance().get_program(), "relu_forward"),
    Relu_backward_kernel(opencl_runtime::getInstance().get_program(), "relu_backward"),
    // �b�o�̪�����l�� cl::EnqueueArgs ����
    enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size)),
    enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size))
{
    // �c�y����餺�{�b�i�H�O���ťաA�Ϊ̰����L���A�Φ�����Ȫ��޿�
}
Relu::~Relu() {
	// �M�z�귽
}
std::string Relu::get_name() {
	return "relu";
}
void Relu::Get_Tensor(Tensor& output) {
	output = Tensor({ input_size });
}
void Relu::forward(const Tensor& input, Tensor& output) {
	// �ϥ� OpenCL ���ֶi��e�V�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Relu_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), input_size);
}
void Relu::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �ϥ� OpenCL ���ֶi��ϦV�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Relu_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), input.get_buffer(), grad_input.get_buffer(), input_size);
}


Softmax::Softmax(int input_size)
	: input_size(input_size), // ��l�� total_elements_
	// �b�o�̪�����l�� cl::make_kernel ����
	Softmax_forward_kernel(opencl_runtime::getInstance().get_program(), "softmax_forward"),
	Softmax_backward_kernel(opencl_runtime::getInstance().get_program(), "softmax_backward"),
	// �b�o�̪�����l�� cl::EnqueueArgs ����
	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size)),
	enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_size))
{
	// �c�y����餺�{�b�i�H�O���ťաA�Ϊ̰����L���A�Φ�����Ȫ��޿�
}
Softmax::~Softmax() {
	// �M�z�귽
}
std::string Softmax::get_name() {
	return "softmax";
}
void Softmax::Get_Tensor(Tensor& output) {
	output = Tensor({ input_size });
}
void Softmax::forward(const Tensor& input, Tensor& output) {
	// �ϥ� OpenCL ���ֶi��e�V�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Softmax_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), input_size);
	// �O�s��X�H�K�b�ϦV�Ǽ����ϥ�
	output_backup = output.get_buffer(); // �O�s��X�� output_backup
}
void Softmax::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �ϥ� OpenCL ���ֶi��ϦV�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Softmax_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), output_backup, grad_input.get_buffer(), input_size);
}


Dense::Dense(int input_size, int output_size)
	: input_size(input_size), output_size(output_size),
	// �b�o�̪�����l�� cl::make_kernel ����
	Dense_forward_kernel(opencl_runtime::getInstance().get_program(), "dense_forward"),
	Dense_backward_wb_kernel(opencl_runtime::getInstance().get_program(), "dense_backward_wb"),
	Dense_backward_input_kernel(opencl_runtime::getInstance().get_program(), "dense_backward_input"),
	update_vector_kernel(opencl_runtime::getInstance().get_program(), "update_vector"),
	// �b�o�̪�����l�� cl::EnqueueArgs ����
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
	// �M�z�귽
}
std::string Dense::get_name() {
	return "dense";
}
void Dense::Get_Tensor(Tensor& output) {
	output = Tensor({ output_size });
}
void Dense::forward(const Tensor& input, Tensor& output) {
	// �ϥ� OpenCL ���ֶi��e�V�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Dense_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), weights.get_buffer(), biases.get_buffer(), input_size, output_size);
}
void Dense::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �ϥ� OpenCL ���ֶi��ϦV�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Dense_backward_wb_kernel(enqueue_args_backward_wb, grad_output.get_buffer(), input.get_buffer(), grad_weights.get_buffer(), grad_biases.get_buffer(), input_size, output_size);
	Dense_backward_input_kernel(enqueue_args_backward_input, grad_output.get_buffer(), weights.get_buffer(), grad_input.get_buffer(), input_size, output_size);
}


Convolution::Convolution(int input_channels, int input_size, int output_channels, int kernel_size)
	: input_channels(input_channels), input_size(input_size), output_channels(output_channels), kernel_size(kernel_size),
	// �b�o�̪�����l�� cl::make_kernel ����
	Convolution_forward_kernel(opencl_runtime::getInstance().get_program(), "convolution_forward"),
	Convolution_backward_weights_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_weights"),
	Convolution_backward_biases_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_biases"),
	Convolution_backward_input_kernel(opencl_runtime::getInstance().get_program(), "convolution_backward_input"),
	update_vector_kernel(opencl_runtime::getInstance().get_program(), "update_vector"),
	// �b�o�̪�����l�� cl::EnqueueArgs ����
	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(output_channels, input_size - kernel_size + 1, input_size - kernel_size + 1)),
	enqueue_args_backward_weights(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, kernel_size, kernel_size)),
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
	grad_weights.transfer_to_gpu();
	grad_biases.transfer_to_gpu();
}
Convolution::~Convolution() {
	// �M�z�귽
}
std::string Convolution::get_name() {
	return "convolution";
}
void Convolution::Get_Tensor(Tensor& output) {
	output = Tensor({ output_channels, input_size - kernel_size + 1, input_size - kernel_size + 1 });
}
void Convolution::forward(const Tensor& input, Tensor& output) {
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	// �ϥ� OpenCL ���ֶi��e�V�Ǽ�
	Convolution_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), weights.get_buffer(), biases.get_buffer(), input_channels, input_size, output_channels, kernel_size);
}
void Convolution::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �ϥ� OpenCL ���ֶi��ϦV�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	for (int h = 0; h < output_channels; ++h) {
		Convolution_backward_weights_kernel(enqueue_args_backward_weights, grad_output.get_buffer(), input.get_buffer(), grad_weights.get_buffer(), h, input_channels, input_size, kernel_size);
	}
	Convolution_backward_biases_kernel(enqueue_args_backward_biases, grad_output.get_buffer(), grad_biases.get_buffer(), output_channels, input_size - kernel_size + 1);
	Convolution_backward_input_kernel(enqueue_args_backward_input, grad_output.get_buffer(), weights.get_buffer(), grad_input.get_buffer(), input_channels, input_size, output_channels, kernel_size);
}


Pooling::Pooling(int input_channels, int input_size, int pool_size)
	: input_channels(input_channels), input_size(input_size), pool_size(pool_size),
	// �b�o�̪�����l�� cl::make_kernel ����
	Pooling_forward_kernel(opencl_runtime::getInstance().get_program(), "pooling_forward"),
	Pooling_backward_kernel(opencl_runtime::getInstance().get_program(), "pooling_backward"),
	// �b�o�̪�����l�� cl::EnqueueArgs ����
	enqueue_args_forward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, (input_size / pool_size), (input_size / pool_size))),
	enqueue_args_backward(opencl_runtime::getInstance().get_queue(), cl::NDRange(input_channels, (input_size / pool_size), (input_size / pool_size)))
{
	int output_dim = input_size / pool_size;
	if (output_dim <= 0) {
		throw std::invalid_argument("Invalid input_size or pool_size for pooling layer.");
	}
	max_indices = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, sizeof(int) * input_channels * output_dim * output_dim * 2);

}
Pooling::~Pooling() {
	// �M�z�귽
}
std::string Pooling::get_name() {
	return "pooling";
}
void Pooling::Get_Tensor(Tensor& output) {
	int output_dim = input_size / pool_size;
	output = Tensor({ input_channels, output_dim, output_dim });
}
void Pooling::forward(const Tensor& input, Tensor& output) {
	// �ϥ� OpenCL ���ֶi��e�V�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Pooling_forward_kernel(enqueue_args_forward, input.get_buffer(), output.get_buffer(), max_indices, input_channels, input_size, pool_size);
}
void Pooling::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �ϥ� OpenCL ���ֶi��ϦV�Ǽ�
	opencl_runtime::getInstance().get_queue().finish(); // �T�O�e�@�Ӿާ@����
	Pooling_backward_kernel(enqueue_args_backward, grad_output.get_buffer(), grad_input.get_buffer(), max_indices, input_channels, input_size, pool_size);
}


Flatten_3D::Flatten_3D(int input_channels, int input_size)
	: input_channels(input_channels), input_size(input_size)
{ 
	// Flatten_3D �h���ݭn OpenCL ���֩� EnqueueArgs�A�]�����u�O�N�h���i�q�i�����@���i�q
}
Flatten_3D::~Flatten_3D() {
	// �M�z�귽
}
std::string Flatten_3D::get_name() {
	return "flatten_3d";
}
void Flatten_3D::Get_Tensor(Tensor& output) {
	int total_elements = input_channels * input_size * input_size;
	output = Tensor({ total_elements });
}
void Flatten_3D::forward(const Tensor& input, Tensor& output) {
	// �N 3D �i�q�i���� 1D �i�q

	output.copy_from(input);
}
void Flatten_3D::backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
	// �N 1D ��ױi�q�i���� 3D �i�q
	grad_input.copy_from(grad_output);
}

	
