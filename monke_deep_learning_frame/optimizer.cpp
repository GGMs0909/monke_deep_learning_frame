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
    // 初始化 Adam Kernel
    adam_kernel(opencl_runtime::getInstance().get_program(), "adam_update"),
    gradient_reset_kernel(opencl_runtime::getInstance().get_program(), "reset_gradient") {
    // m 和 v 向量將在 initialize_moments 中初始化
}

Adam::~Adam() {
    // m 和 v 是 std::vector<Tensor>，它們的 Tensor 解構子會自動清理 cl::Buffer
}

void Adam::initialize_moments(const std::vector<Tensor*>& parameters) {
    if (!m.empty() || !v.empty()) {
        std::cerr << "Warning: Adam moments already initialized. Re-initializing." << std::endl;
        m.clear();
        v.clear();
    }

    m.reserve(parameters.size());
    v.reserve(parameters.size());

    // 創建一個填充了零的 CPU 端數據向量
    // 注意：這會為每個 Tensor 都創建一個新的零向量，如果參數非常大，可能會佔用較多 CPU 記憶體
    for (const auto& param_ptr : parameters) {
        size_t param_total_size = param_ptr->size(); // 獲取參數的總元素數量
   // 創建一個充滿零的浮點向量

        // 使用 CPU 端的零向量來初始化 m 和 v 的 Tensor
        // 這會觸發 Tensor 內部 cl::Buffer 的創建和初始化數據傳輸到 GPU
        Tensor m_tensor(param_ptr->shape); // 假設 Tensor 建構子支援 (Shape, std::vector<float>)
        // m_tensor.transfer_to_gpu(); // 如果建構子已經傳輸，則不需要額外呼叫

        Tensor v_tensor(param_ptr->shape);
        // v_tensor.transfer_to_gpu(); // 如果建構子已經傳輸，則不需要額外呼叫

        m.push_back(std::move(m_tensor)); // 使用移動語義將 Tensor 加入向量
        v.push_back(std::move(v_tensor)); // 使用移動語義
    }
    t = 0; // 重置時間步
    opencl_runtime::getInstance().get_queue().finish(); // 確保所有 m 和 v 的 GPU 端初始化完成
    std::cout << "Adam moments initialized for " << parameters.size() << " parameters." << std::endl;
}
void Adam::update(std::vector<Tensor*> parameters, std::vector<Tensor*> grad_parameters) {
    if (m.empty() || v.empty() || m.size() != parameters.size()) {
        throw std::runtime_error("Adam moments (m, v) not initialized. Call initialize_moments() first.");
    }

    t++; // 增加時間步計數器

    cl::CommandQueue& queue = opencl_runtime::getInstance().get_queue();

    for (size_t i = 0; i < parameters.size(); ++i) {

        // 確保所有相關 Tensor 都在 GPU 上

        // 呼叫 Adam 更新 Kernel
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
            t, // 傳遞當前時間步 t
            (int)parameters[i]->size());
    }
    queue.finish(); // 確保所有更新操作完成
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