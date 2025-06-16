//optimizer.cpp
#include "optimizer.h"
#include <stdexcept>

GradientDescent::GradientDescent() : 
    kernel(opencl_runtime::getInstance().get_program(), "gradient_decent"), 
    gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "reset_gradient")  {
	// Initialize
}
GradientDescent::~GradientDescent() {
	// Clean up resources if needed
}
void GradientDescent::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) {
	// Update weights using OpenCL kernel
	for (int i = 0; i < parameters.size(); ++i) {
		if (parameters[i]->size() != grad_parameters[i]->size()) {
			throw std::invalid_argument("Parameter and gradient sizes do not match.");
		}
        opencl_runtime::getInstance().get_queue().finish();
		kernel(cl::EnqueueArgs(opencl_runtime::getInstance().get_queue(), cl::NDRange(parameters[i]->size())),
			parameters[i]->get_buffer(), grad_parameters[i]->get_buffer(), learning_rate, parameters[i]->size());
	}
	// Reset gradients done in kernel
}
void GradientDescent::reset(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
    for (int i = 0; i < parameters.size(); ++i) {
		if (parameters[i]->size() != grad_parameters[i]->size()) {
			throw std::invalid_argument("Parameter and gradient sizes do not match.");
		}
		// Reset gradients using OpenCL kernel
        opencl_runtime::getInstance().get_queue().finish();
		gradient_reset_kernel(cl::EnqueueArgs(opencl_runtime::getInstance().get_queue(), cl::NDRange(grad_parameters[i]->size())),
			grad_parameters[i]->get_buffer(), grad_parameters[i]->size());
    }
}

Adam::Adam(float beta1, float beta2, float epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0),

    adam_kernel(opencl_runtime::getInstance().get_program(), "adam_update"),
    gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "reset_gradient") {

}

Adam::~Adam() {

}

void Adam::initialize_moments(const std::vector<Tensor*>& parameters) {
    if (!m.empty() || !v.empty()) {
        std::cerr << "Warning: Adam moments already initialized. Re-initializing." << std::endl;
        m.clear();
        v.clear();
    }

    m.reserve(parameters.size());
    v.reserve(parameters.size());


    for (const auto& param_ptr : parameters) {
        size_t param_total_size = param_ptr->size();



        Tensor m_tensor(param_ptr->shape); 


        Tensor v_tensor(param_ptr->shape);


        m.push_back(std::move(m_tensor));
        v.push_back(std::move(v_tensor));
    }
    t = 0;
    opencl_runtime::getInstance().get_queue().finish(); 
    std::cout << "Adam moments initialized for " << parameters.size() << " parameters." << std::endl;
}
void Adam::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) {
    if (m.empty() || v.empty() || m.size() != parameters.size()) {
        throw std::runtime_error("Adam moments (m, v) not initialized. Call initialize_moments() first.");
    }

    t++; 

    cl::CommandQueue& queue = opencl_runtime::getInstance().get_queue();

    for (size_t i = 0; i < parameters.size(); ++i) {


        // Args: param_buffer, grad_buffer, m_buffer, v_buffer, learning_rate, beta1, beta2, epsilon, t, size
        opencl_runtime::getInstance().get_queue().finish();
        adam_kernel(cl::EnqueueArgs(queue, cl::NDRange(parameters[i]->size())),
            parameters[i]->get_buffer(),
            grad_parameters[i]->get_buffer(),
            m[i].get_buffer(),
            v[i].get_buffer(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t,
            (int)parameters[i]->size());
    }
    queue.finish();
}
void Adam::reset(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
    for (int i = 0; i < parameters.size(); ++i) {
        if (parameters[i]->size() != grad_parameters[i]->size()) {
            throw std::invalid_argument("Parameter and gradient sizes do not match.");
        }
        // Reset gradients using OpenCL kernel
        opencl_runtime::getInstance().get_queue().finish();
        gradient_reset_kernel(cl::EnqueueArgs(opencl_runtime::getInstance().get_queue(), cl::NDRange(grad_parameters[i]->size())),
            grad_parameters[i]->get_buffer(), grad_parameters[i]->size());
    }
}