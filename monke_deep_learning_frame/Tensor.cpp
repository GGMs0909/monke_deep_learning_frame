#include "Tensor.h"
#include <numeric> // std::accumulate
#include <algorithm> // std::min
#include <sstream> // std::stringstream

// 建構子：根據形狀初始化 Tensor
Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    calculate_strides();
    data = std::vector<float>(calculate_total_size(),0);
}

// 建構子：從現有資料和形狀建立 Tensor (複製資料)
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape(shape), data(data) {
    if (data.size() != calculate_total_size()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    calculate_strides();
}

// 取得 Tensor 的總元素數量
size_t Tensor::size() const {
    return data.size();
}

// 根據索引取得元素 (const 版本)
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

// 根據索引取得元素的引用 (可修改 Tensor 內容)
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

// 取得指向內部資料的指標 (謹慎使用)
float* Tensor::data_ptr() {
    return data.data();
}

const float* Tensor::data_ptr() const {
    return data.data();
}

// 打印 Tensor 的形狀和部分內容 (方便調試)
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

// 計算總元素數量
size_t Tensor::calculate_total_size() const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// 計算步長
void Tensor::calculate_strides() {
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// 將多維索引轉換為一維線性索引 (使用步長)
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