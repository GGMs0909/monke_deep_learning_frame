#include "tensor.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>

void Tensor::calculateStrides() {
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

inline size_t Tensor::getFlatIndex(const std::vector<size_t>& indices){
    if(indices.size() != shape.size()){
        throw std::out_of_range("Number of indices does not match number of dimensions.");
    }
    size_t result = 0;
    for(int i = 0; i < indices.size(); i++){
        if(indices[i] >= shape[i]){
            throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
        }
        result += strides[i] * indices[i];
    }
    return result;
}

Tensor::Tensor(const std::vector<size_t>& shape_) : shape(shape_), on_gpu(false) {
    calculateStrides();
    total_size = 1;
    for (const auto& dim : shape) {
        total_size *= dim;
    }
    sizebytes = total_size * sizeof(float);
    data.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape_, const std::vector<float>& data_) 
    : shape(shape_), data(data_), on_gpu(false) {
    calculateStrides();
    total_size = 1;
    for (const auto& dim : shape) {
        total_size *= dim;
    }
    if(data.size() != total_size){
        throw std::invalid_argument("Data size does not match tensor shape.");
    }
    sizebytes = total_size * sizeof(float);
}

Tensor::~Tensor(){
    //a
}

size_t Tensor::getTotalSize() {
    return total_size;
}

void Tensor::toGPU() const {
    if (!on_gpu) {
        cl::Context& context = opencl_runtime::getInstance().get_context();
        buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizebytes, data.data());
        on_gpu = true;
    }
    else {
        throw std::runtime_error("Data is already on GPU.");
    }
}
void Tensor::toCPU() const {
    if (on_gpu) {
        cl::CommandQueue& queue = opencl_runtime::getInstance().get_queue();
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizebytes, data.data());
        on_gpu = false;
    }
    else {
        throw std::runtime_error("Data is already on CPU.");
    }
}

void Tensor::toGPU() {
    if (!on_gpu) {
        cl::Context& context = opencl_runtime::getInstance().get_context();
        buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizebytes, data.data());
        on_gpu = true;
    }
    else {
        throw std::runtime_error("Data is already on GPU.");
    }
}
void Tensor::toCPU() {
    if (on_gpu) {
        cl::CommandQueue& queue = opencl_runtime::getInstance().get_queue();
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizebytes, data.data());
        on_gpu = false;
    }
    else {
        throw std::runtime_error("Data is already on CPU.");
    }
}

std::vector<float> Tensor::getData() const {
    if (on_gpu) {
        throw std::runtime_error("(from Tensor::getData())Data is on GPU. Call toCPU() before accessing data.");
    }
    return data;
}
std::vector<float>& Tensor::getDataRef() {
    if (on_gpu) {
        throw std::runtime_error("(from Tensor::getDataRef())Data is on GPU. Call toCPU() before accessing data.");
    }
    return data;
}
cl::Buffer Tensor::getBuffer() {
    if (!on_gpu) {
        throw std::runtime_error("Buffer is not on GPU.");
    }
    return buffer;
}
cl::Buffer Tensor::getBuffer() const {
    if (!on_gpu) {
        throw std::runtime_error("Buffer is not on GPU.");
    }
    return buffer;
}
std::vector<size_t> Tensor::getShape() const {
    return shape;
}

void Tensor::set(const std::vector<size_t>& indices, float value) {
    if (on_gpu) {
        throw std::runtime_error("Data is on GPU. Call toCPU() before setting data.");
    }
    size_t flat_index = getFlatIndex(indices);
    data[flat_index] = value;
}

float Tensor::get(const std::vector<size_t>& indices) {
    if (on_gpu) {
        throw std::runtime_error("Data is on GPU. Call toCPU() before getting data.");
    }
    size_t flat_index = getFlatIndex(indices);
    return data[flat_index];
}

void Tensor::print(const size_t max_elements) {
    if (on_gpu) {
        throw std::runtime_error("Data is on GPU. Call toCPU() before printing data.");
    }
    size_t elements_to_print = std::min<size_t>(total_size, max_elements);
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < elements_to_print; i++) {
        std::cout << data[i];
        if (i < elements_to_print - 1) std::cout << ", ";
    }
    if (total_size > max_elements) {
        std::cout << ", ...";
    }
    std::cout << "])" << std::endl;
}

void combine_tensors(const std::vector<Tensor*>& tensors, Tensor& output) {
    if (tensors.empty()) return;

    // 1. 安全檢查：確保所有 Tensor 的形狀一致
    std::vector<size_t> base_shape = tensors[0]->getShape();
    size_t single_tensor_size = tensors[0]->getTotalSize();
    
    // 2. 準備 CPU 端的匯總空間 (getDataRef 獲取 output 的資料引用)
    std::vector<float>& output_data = output.getDataRef();
    output_data.clear();
    output_data.reserve(single_tensor_size * tensors.size());

    // 3. 逐個拷貝資料 (這裡是 CPU 端的拼接)
    for (size_t i = 0; i < tensors.size(); ++i) {
        // 確保形狀匹配，避免 Batch 裡出現形狀怪異的資料
        if (tensors[i]->getShape() != base_shape) {
            throw std::runtime_error("Tensor shape mismatch in combine_tensors!");
        }
        
        // 獲取各個小 Tensor 的資料
        std::vector<float> vec = tensors[i]->getData(); 
        output_data.insert(output_data.end(), vec.begin(), vec.end());
    }
}