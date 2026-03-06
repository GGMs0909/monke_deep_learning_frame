#include "layer.hpp"
#include <random>


static float random_normal_float(float mean, float stddev) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<float> dist(mean, stddev);
	return dist(gen);
}
static int generate_seed() {
	static std::random_device rd;
	return rd(); 
}


//scale
Scale::Scale(size_t size_, float scale_factor_) :
    size(size_),
    scale_factor(scale_factor_),
    forward_kernel(
        opencl_runtime::getInstance().get_program(), "scale_forward"
    ),
    backward_kernel(
        opencl_runtime::getInstance().get_program(), "scale_backward"
    )
{}

void Scale::predict(const Tensor& inputs, Tensor& outputs) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        size,
        scale_factor
    );
    opencl_runtime::getInstance().get_queue().finish();
}

void Scale::forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        size*batch_size,
        scale_factor
    );
    opencl_runtime::getInstance().get_queue().finish();
}

void Scale::backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) {
    backward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        grad_outputs.getBuffer(),
        grad_inputs.getBuffer(),
        size*batch_size,
        scale_factor
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Scale::pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    // No learnable parameters for Scale layer
}
void Scale::get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) {
    if(batch_size == 0) {
        inputs.push_back(new Tensor({(size_t)size}));
        return;
    }
    inputs.push_back(new Tensor({batch_size, (size_t)size}));
}
std::string Scale::get_name() const {
    return "Scale " + std::to_string(size) + " with factor " + std::to_string(scale_factor);
}


//dense
Dense::Dense(size_t input_size_, size_t output_size_) :
    input_size(input_size_),
    output_size(output_size_),
    weights({output_size_, input_size_}),
    biases({output_size_}),
    grad_weights({output_size_, input_size_}),
    grad_biases({output_size_}),
    forward_kernel(
        opencl_runtime::getInstance().get_program(), "dense_forward2"
    ),
    backward_wb_kernel(
        opencl_runtime::getInstance().get_program(), "dense_backward_wb"
    ),
    backward_input_kernel(
        opencl_runtime::getInstance().get_program(), "dense_backward_inputs"
    )

{
    for(size_t i = 0; i < output_size; i++) {
        for(size_t j = 0; j < input_size; j++){
            weights.set({i,j}, random_normal_float(0.0f, 1.0f / std::sqrt(input_size)));
        }
        biases.set({i}, 0.0f);
    }
    
    weights.toGPU();
    biases.toGPU();
    grad_weights.toGPU();
    grad_biases.toGPU();
}
void Dense::predict(const Tensor& inputs, Tensor& outputs) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange((output_size+15)/16*16, 16),
            cl::NDRange(16,16)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        weights.getBuffer(),
        biases.getBuffer(),
        1,
        input_size,
        output_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Dense::forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange((output_size+15)/16*16, ((batch_size+15)/16)*16),
            cl::NDRange(16,16)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        weights.getBuffer(),
        biases.getBuffer(),
        static_cast<int>(batch_size),
        input_size,
        output_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Dense::backward(
    const Tensor& output_grads,
    const Tensor& inputs,
    Tensor& input_grads,
    size_t batch_size
) {
    // Compute gradients w.r.t. weights and biases
    backward_wb_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(input_size, output_size)
        ),
        output_grads.getBuffer(),
        inputs.getBuffer(),
        grad_weights.getBuffer(),
        grad_biases.getBuffer(),
        static_cast<int>(batch_size),
        output_size,
        input_size
    );
    opencl_runtime::getInstance().get_queue().finish();

    // Compute gradients w.r.t. inputs
    backward_input_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(input_size, batch_size)
        ),
        output_grads.getBuffer(),
        weights.getBuffer(),
        input_grads.getBuffer(),
        static_cast<int>(batch_size),
        output_size,
        input_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Dense::pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads)
{
    params.push_back(&weights);
    params.push_back(&biases);
    grads.push_back(&grad_weights);
    grads.push_back(&grad_biases);
}
void Dense::get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) {
    if(batch_size == 0) {
        inputs.push_back(new Tensor({(size_t)output_size}));
        return;
    }
    inputs.push_back(new Tensor({batch_size, (size_t)output_size}));
}

std::string Dense::get_name() const {
    return "Dense " + std::to_string(input_size) + " -> " + std::to_string(output_size);
}


//convolution
Convolution::Convolution(size_t input_channels_, size_t input_size_, size_t output_channels_, size_t kernel_size_) :
    input_channels(input_channels_),
    input_size(input_size_),
    output_channels(output_channels_),
    kernel_size(kernel_size_),
    weights({output_channels_, input_channels_, kernel_size_, kernel_size_}),
    biases({output_channels_}),
    grad_weights({output_channels_, input_channels_, kernel_size_, kernel_size_}),
    grad_biases({output_channels_}),
    forward_kernel(
        opencl_runtime::getInstance().get_program(), "convolution_forward"
    ),
    backward_weight_kernel(
        opencl_runtime::getInstance().get_program(), "convolution_backward_weights"
    ),
    backward_bias_kernel(
        opencl_runtime::getInstance().get_program(), "convolution_backward_biases"
    ),
    backward_input_kernel(
        opencl_runtime::getInstance().get_program(), "convolution_backward_inputs"
    )
{
    for(size_t oc = 0; oc < output_channels; oc++) {
        for(size_t ic = 0; ic < input_channels; ic++) {
            for(size_t kx = 0; kx < kernel_size; kx++) {
                for(size_t ky = 0; ky < kernel_size; ky++) {
                    weights.set({oc, ic, kx, ky}, random_normal_float(0.0f, 1.0f / std::sqrt(input_channels * kernel_size * kernel_size)));
                }
            }
        }
        biases.set({oc}, 0.0f);
    }
    
    weights.toGPU();
    biases.toGPU();
    grad_weights.toGPU();
    grad_biases.toGPU();
}
void Convolution::predict(const Tensor& inputs, Tensor& outputs) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange((input_size-kernel_size+1)*(input_size-kernel_size+1),output_channels, 1)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        weights.getBuffer(),
        biases.getBuffer(),
        1,
        input_channels,
        input_size,
        output_channels,
        (input_size-kernel_size+1),
        kernel_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Convolution::forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange((input_size-kernel_size+1)*(input_size-kernel_size+1),output_channels, batch_size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        weights.getBuffer(),
        biases.getBuffer(),
        batch_size,
        input_channels,
        input_size,
        output_channels,
        (input_size-kernel_size+1),
        kernel_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Convolution::backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) {
    // Compute gradients w.r.t. weights
    backward_weight_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(input_channels*kernel_size*kernel_size, output_channels)
        ),
        grad_outputs.getBuffer(),
        inputs.getBuffer(),
        grad_weights.getBuffer(),
        (int)(batch_size),
        input_channels,
        input_size,
        output_channels,
        input_size-kernel_size+1,
        kernel_size
    );
    opencl_runtime::getInstance().get_queue().finish();

    // Compute gradients w.r.t. biases
    backward_bias_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(output_channels)
        ),
        grad_outputs.getBuffer(),
        grad_biases.getBuffer(),
        static_cast<int>(batch_size),
        output_channels,
        input_size-kernel_size+1
    );
    opencl_runtime::getInstance().get_queue().finish();

    // Compute gradients w.r.t. inputs
    backward_input_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(input_size*input_size, input_channels, batch_size)
        ),
        grad_outputs.getBuffer(),
        weights.getBuffer(),
        grad_inputs.getBuffer(),
        (int)batch_size,
        input_channels,
        input_size,
        output_channels,
        input_size-kernel_size+1,
        kernel_size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Convolution::pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    params.push_back(&weights);
    params.push_back(&biases);
    grads.push_back(&grad_weights);
    grads.push_back(&grad_biases);
}
void Convolution::get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) {
    if(batch_size == 0) {
        inputs.push_back(new Tensor({(size_t)output_channels, (size_t)(input_size - kernel_size + 1), (size_t)(input_size - kernel_size + 1)}));
        return;
    }
    inputs.push_back(new Tensor({batch_size, (size_t)output_channels, (size_t)(input_size - kernel_size + 1), (size_t)(input_size - kernel_size + 1)}));
}
std::string Convolution::get_name() const {
    return "Convolution " + std::to_string(input_channels) + "x" + std::to_string(input_size) + "x" + std::to_string(input_size) + " -> " + std::to_string(output_channels) + "x" + std::to_string(input_size - kernel_size + 1) + "x" + std::to_string(input_size - kernel_size + 1);
}


//relu
ReLU::ReLU(size_t size_, float slope_) :
    size(size_),
    slope(slope_),
    relu_forward_kernel(
        opencl_runtime::getInstance().get_program(), "relu_forward"
    ),
    relu_backward_kernel(
        opencl_runtime::getInstance().get_program(), "relu_backward"
    )
{}
void ReLU::predict(const Tensor& inputs, Tensor& outputs){
    relu_forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        size,
        slope
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void ReLU::forward(const Tensor& inputs, Tensor& outputs, size_t batch_size){
    relu_forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        size*batch_size,
        slope
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void ReLU::backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size){
    relu_backward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        grad_outputs.getBuffer(),
        inputs.getBuffer(),
        grad_inputs.getBuffer(),
        size*batch_size,
        slope
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void ReLU::pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads){
    // ReLU has no parameters
}
void ReLU::get_tensor(std::vector<Tensor*>& inputs, size_t batch_size){
    if(batch_size == 0) {
        inputs.push_back(new Tensor({(size_t)size}));
        return;
    }
    inputs.push_back(new Tensor({batch_size, (size_t)size}));
}
std::string ReLU::get_name() const {
    return "ReLU " + std::to_string(size) + " (slope=" + std::to_string(slope) + ")";
}
//softmax
Softmax::Softmax(size_t size_) :
    size(size_),
    forward_kernel(
        opencl_runtime::getInstance().get_program(), "softmax_forward"
    ),
    backward_kernel(
        opencl_runtime::getInstance().get_program(), "softmax_backward"
    )
{}
void Softmax::predict(const Tensor& inputs, Tensor& outputs) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size,1)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        1,
        size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Softmax::forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) {
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size, batch_size)
        ),
        inputs.getBuffer(),
        outputs.getBuffer(),
        batch_size,
        size
    );
    outputs_backup = outputs.getBuffer(); // Backup for backward pass
    opencl_runtime::getInstance().get_queue().finish();
}
void Softmax::backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) {
    backward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size, batch_size)
        ),
        grad_outputs.getBuffer(),
        outputs_backup,
        grad_inputs.getBuffer(),
        batch_size,
        size
    );
    opencl_runtime::getInstance().get_queue().finish();
}
void Softmax::pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    // Softmax has no parameters
}
void Softmax::get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) {
    if(batch_size == 0) {
        inputs.push_back(new Tensor({(size_t)size}));
        return;
    }
    
    inputs.push_back(new Tensor({batch_size, (size_t)size}));
}
std::string Softmax::get_name() const {
    return "Softmax " + std::to_string(size);
}
