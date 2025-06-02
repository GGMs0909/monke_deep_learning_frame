//optimizer.cpp
#include "optimizer.h"
#include <stdexcept>

GradientDescent::GradientDescent(float learning_rate) : 
    learning_rate(learning_rate), 
    kernel(opencl_runtime::getInstance().get_program(), "gradient_decent"), 
    gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "reset_gradient")  {
	// Initialize
}
GradientDescent::~GradientDescent() {
	// Clean up resources if needed
}
void GradientDescent::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
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

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0),
    // ��l�� Adam Kernel
    adam_kernel(opencl_runtime::getInstance().get_program(), "adam_update"),
    gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "reset_gradient") {
    // m �M v �V�q�N�b initialize_moments ����l��
}

Adam::~Adam() {
    // m �M v �O std::vector<Tensor>�A���̪� Tensor �Ѻc�l�|�۰ʲM�z cl::Buffer
}

void Adam::initialize_moments(const std::vector<Tensor*>& parameters) {
    if (!m.empty() || !v.empty()) {
        std::cerr << "Warning: Adam moments already initialized. Re-initializing." << std::endl;
        m.clear();
        v.clear();
    }

    m.reserve(parameters.size());
    v.reserve(parameters.size());

    // �Ыؤ@�Ӷ�R�F�s�� CPU �ݼƾڦV�q
    // �`�N�G�o�|���C�� Tensor ���Ыؤ@�ӷs���s�V�q�A�p�G�ѼƫD�`�j�A�i��|���θ��h CPU �O����
    for (const auto& param_ptr : parameters) {
        size_t param_total_size = param_ptr->size(); // ����Ѽƪ��`�����ƶq
   // �Ыؤ@�ӥR���s���B�I�V�q

        // �ϥ� CPU �ݪ��s�V�q�Ӫ�l�� m �M v �� Tensor
        // �o�|Ĳ�o Tensor ���� cl::Buffer ���ЫةM��l�Ƽƾڶǿ�� GPU
        Tensor m_tensor(param_ptr->shape); // ���] Tensor �غc�l�䴩 (Shape, std::vector<float>)
        // m_tensor.transfer_to_gpu(); // �p�G�غc�l�w�g�ǿ�A�h���ݭn�B�~�I�s

        Tensor v_tensor(param_ptr->shape);
        // v_tensor.transfer_to_gpu(); // �p�G�غc�l�w�g�ǿ�A�h���ݭn�B�~�I�s

        m.push_back(std::move(m_tensor)); // �ϥβ��ʻy�q�N Tensor �[�J�V�q
        v.push_back(std::move(v_tensor)); // �ϥβ��ʻy�q
    }
    t = 0; // ���m�ɶ��B
    opencl_runtime::getInstance().get_queue().finish(); // �T�O�Ҧ� m �M v �� GPU �ݪ�l�Ƨ���
    std::cout << "Adam moments initialized for " << parameters.size() << " parameters." << std::endl;
}
void Adam::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
    if (m.empty() || v.empty() || m.size() != parameters.size()) {
        throw std::runtime_error("Adam moments (m, v) not initialized. Call initialize_moments() first.");
    }

    t++; // �W�[�ɶ��B�p�ƾ�

    cl::CommandQueue& queue = opencl_runtime::getInstance().get_queue();

    for (size_t i = 0; i < parameters.size(); ++i) {

        // �T�O�Ҧ����� Tensor ���b GPU �W

        // �I�s Adam ��s Kernel
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
            t, // �ǻ���e�ɶ��B t
            (int)parameters[i]->size());
    }
    queue.finish(); // �T�O�Ҧ���s�ާ@����
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