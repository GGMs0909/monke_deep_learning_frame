#include "optimizer.hpp"
#include <cmath>

// GD
GD::GD() : gd_update_kernel(opencl_runtime::getInstance().get_program(), "gd_update"), zero_grad_kernel(opencl_runtime::getInstance().get_program(), "set_to_zero") {}
void GD::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) {
    for(int i = 0; i < parameters.size(); i++){
        gd_update_kernel(
            cl::EnqueueArgs(
                opencl_runtime::getInstance().get_queue(),
                cl::NDRange(parameters[i]->getTotalSize())
            ),
            parameters[i]->getBuffer(),
            grad_parameters[i]->getBuffer(),
            learning_rate,
            parameters[i]->getTotalSize()
        );
        zero_grad_kernel(
            cl::EnqueueArgs(
                opencl_runtime::getInstance().get_queue(),
                cl::NDRange(grad_parameters[i]->getTotalSize())
            ),
            grad_parameters[i]->getBuffer(),
            grad_parameters[i]->getTotalSize()
        );
    }
}

// adam
Adam::Adam(float beta1_, float beta2_, float epsilon_) : beta1(beta1_), beta2(beta2_), epsilon(epsilon_), t(0), adam_update_kernel(opencl_runtime::getInstance().get_program(), "adam_update"), zero_grad_kernel(opencl_runtime::getInstance().get_program(), "set_to_zero") {}
void Adam::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters, float learning_rate) {
    t++;
    for(int i = 0; i < parameters.size(); i++){
        adam_update_kernel(
            cl::EnqueueArgs(
                opencl_runtime::getInstance().get_queue(),
                cl::NDRange(parameters[i]->getTotalSize())
            ),
            parameters[i]->getBuffer(),
            grad_parameters[i]->getBuffer(),
            m[i]->getBuffer(),
            v[i]->getBuffer(),
            beta1,
            beta2,
            epsilon,
            std::pow(beta1,t),
            std::pow(beta2,t),
            learning_rate,
            parameters[i]->getTotalSize()
        );
        zero_grad_kernel(
            cl::EnqueueArgs(
                opencl_runtime::getInstance().get_queue(),
                cl::NDRange(grad_parameters[i]->getTotalSize())
            ),
            grad_parameters[i]->getBuffer(),
            grad_parameters[i]->getTotalSize()
        );
    }
}

void Adam::initialize_momentum(std::vector<Tensor*> parameters) {
    m.clear();
    v.clear();
    for(auto& param : parameters){
        m.push_back(new Tensor(param->getShape()));
        v.push_back(new Tensor(param->getShape()));
        m.back()->toGPU();
        v.back()->toGPU();
    }
}