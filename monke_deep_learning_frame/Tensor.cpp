#include "Tensor.h"
#include <numeric> // std::accumulate
#include <algorithm> // std::min
#include <sstream> // std::stringstream

// �غc�l�G�ھڧΪ���l�� Tensor
Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    calculate_strides();
    data = std::vector<float>(calculate_total_size(),0);
}

// �غc�l�G�q�{����ƩM�Ϊ��إ� Tensor (�ƻs���)
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape(shape), data(data) {
    if (data.size() != calculate_total_size()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    calculate_strides();
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

// ���o���V������ƪ����� (�ԷV�ϥ�)
float* Tensor::data_ptr() {
    return data.data();
}

const float* Tensor::data_ptr() const {
    return data.data();
}

// ���L Tensor ���Ϊ��M�������e (��K�ո�)
void Tensor::print(int limit) const {
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << "], Data (first " << limit << " elements): [";
    for (size_t i = 0; i < std::min((size_t)limit, data.size()); ++i) {
        std::cout << data[i] << (i == std::min((size_t)limit, data.size()) - 1 ? "" : ", ");
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