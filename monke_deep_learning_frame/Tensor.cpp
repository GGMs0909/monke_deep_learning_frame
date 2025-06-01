#include "Tensor.h"
#include <numeric> // std::accumulate
#include <algorithm>
#include <sstream> // std::stringstream

// �غc�l�G�ھڧΪ���l�� Tensor
Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    calculate_strides();
    data = std::vector<float>(calculate_total_size(),0);
	sizebyte = data.size() * sizeof(float);
	cl_buffer = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, sizebyte);
    if (!cl_buffer()) {
        throw std::runtime_error("Tensor constructor: Failed to create OpenCL buffer (cl_buffer() check failed).");
    }

    // ���հ���@�Ӽg�J�ާ@�A�T�O�w�İϬO�i�Ϊ�
    float temp_zero = 0.0f;
    cl_int write_err = opencl_runtime::getInstance().get_queue().enqueueWriteBuffer(
        cl_buffer,          // �ؼнw�İ�
        CL_TRUE,            // ����g�J (�P�B)
        0,                  // �����q
        sizeof(float),      // �g�J�@�� float
        &temp_zero          // �g�J���ƾ�
    );

    if (write_err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to write to new cl::Buffer in Tensor constructor. Error code: " << write_err << std::endl;
        throw std::runtime_error("Tensor constructor: Created OpenCL buffer is unusable for writing.");
    }
}

// �غc�l�G�q�{����ƩM�Ϊ��إ� Tensor (�ƻs���)
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape(shape), data(data) {
    if (data.size() != calculate_total_size()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    calculate_strides();
	sizebyte = data.size() * sizeof(float);
    cl_buffer = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE, sizebyte);
    if (cl_buffer()) {
    }
    else {
        throw std::runtime_error("Failed to create OpenCL buffer");
    }
	transfer_to_gpu();
}

Tensor::~Tensor() {
	//std::cout << "Tensor destructor called." << std::endl;
}

// ���o Tensor ���`�����ƶq
size_t Tensor::size() const {
    return data.size();
}

// �ھگ��ި��o���� (const ����)
float Tensor::get(const std::vector<int>& index) const {
    /*
    size_t linear_index = get_linear_index(index);
    if (linear_index >= data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[linear_index];
    */
    return data[get_linear_index(index)];
}

// �ھگ��ި��o�������ޥ� (�i�ק� Tensor ���e)
float& Tensor::get(const std::vector<int>& index) {
    /*
    size_t linear_index = get_linear_index(index);
    if (linear_index >= data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[linear_index];
    */
    return data[get_linear_index(index)];
}

// ���o OpenCL �w�İ�
cl::Buffer Tensor::get_buffer() {
	if (!cl_buffer()) {
		throw std::runtime_error("OpenCL buffer is not initialized");
	}
	return cl_buffer;
}

cl::Buffer Tensor::get_buffer() const {
	if (!cl_buffer()) {
		throw std::runtime_error("OpenCL buffer is not initialized");
	}
	return cl_buffer;
}

// ���o���V������ƪ����� (�ԷV�ϥ�)
float* Tensor::data_ptr() {
    return data.data();
}

const float* Tensor::data_ptr() const {
    return data.data();
}

void Tensor::copy_from(const Tensor& other) {

	data = other.data; // �����ƻs���
	sizebyte = data.size() * sizeof(float);

	// �T�O OpenCL �w�İϤj�p���T
	if (cl_buffer()) {
		cl_buffer = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizebyte, data.data());
	}
	else {
		throw std::runtime_error("Failed to create OpenCL buffer");
	}
}

// �N Tensor ��ƶǿ�� GPU (OpenCL)
void Tensor::transfer_to_gpu() {
	if (cl_buffer()) {
		cl_buffer = cl::Buffer(opencl_runtime::getInstance().get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizebyte, data.data());
	}
	else {
		throw std::runtime_error("Failed to create OpenCL buffer");
	}
}
void Tensor::transfer_to_cpu() {
	if (cl_buffer()) {
		opencl_runtime::getInstance().get_queue().enqueueReadBuffer(cl_buffer, CL_TRUE, 0, data.size() * sizeof(float), data.data());
	}
	else {
		throw std::runtime_error("OpenCL buffer is not initialized");
	}
}



// Fix the problematic line in the print method
void Tensor::print(size_t limit) const {
   std::cout << "Shape: [";
   for (size_t i = 0; i < shape.size(); ++i) {
       std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
   }
   size_t elements_to_print = min(limit, data.size());
   std::cout << "], Data (first " << limit << " elements): [";
   for (size_t i = 0; i < elements_to_print; ++i) { // Cast limit to size_t
       std::cout << data[i] << (i == elements_to_print - 1 ? "" : ", ");
   }
   if (data.size() > limit) {
       std::cout << "...]";
   }
   else {
       std::cout << "]";
   }
   std::cout << std::endl;
}

// �p���`�����ƶq
size_t Tensor::calculate_total_size() const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// �p��B��
void Tensor::calculate_strides() {
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// �N�h�������ഫ���@���u�ʯ��� (�ϥΨB��)
inline size_t Tensor::get_linear_index(const std::vector<int>& index) const {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Incorrect number of indices");
    }
    size_t linear_index = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            std::stringstream ss;
            ss << "Index out of bounds for dimension " << i;
            throw std::out_of_range(ss.str());
        }
        linear_index += index[i] * strides[i];
    }
    return linear_index;
}