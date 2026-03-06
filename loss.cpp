#include "loss.hpp"
#include <cmath>



//MSE
MSE::MSE(size_t size_): size(size_),
                        forward_kernel(opencl_runtime::getInstance().get_program(), "MSE_forward"),
                        backward_kernel(opencl_runtime::getInstance().get_program(), "MSE_backward")  {}
float MSE::forward(const Tensor& predictions, const Tensor& targets, size_t batch_size){
    cl::Buffer temp(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, size * batch_size * sizeof(float));
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        predictions.getBuffer(),
        targets.getBuffer(),
        temp,
        size*batch_size
    );
    return reduction(temp, size*batch_size) / (size*batch_size);
}
void MSE::backward(const Tensor& predictions, const Tensor& targets, Tensor& grad_inputs, size_t batch_size){
    backward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        predictions.getBuffer(),
        targets.getBuffer(),
        grad_inputs.getBuffer(),
        size*batch_size
    );
}
//CrossEntropy
CrossEntropy::CrossEntropy(size_t size_): size(size_),
                                          forward_kernel(opencl_runtime::getInstance().get_program(), "crossentropy_forward"),
                                          backward_kernel(opencl_runtime::getInstance().get_program(), "crossentropy_backward") {}
float CrossEntropy::forward(const Tensor& predictions, const Tensor& targets, size_t batch_size){
    cl::Buffer temp(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, size * batch_size * sizeof(float));
    forward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        predictions.getBuffer(),
        targets.getBuffer(),
        temp,
        size*batch_size
    );
    return reduction(temp, size*batch_size) / batch_size;
}
void CrossEntropy::backward(const Tensor& predictions, const Tensor& targets, Tensor& grad_inputs, size_t batch_size){
    backward_kernel(
        cl::EnqueueArgs(
            opencl_runtime::getInstance().get_queue(),
            cl::NDRange(size*batch_size)
        ),
        predictions.getBuffer(),
        targets.getBuffer(),
        grad_inputs.getBuffer(),
        size,
        batch_size
    );
}