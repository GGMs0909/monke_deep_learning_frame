#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include "opencl_runtime.h"


class Tensor {
public:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
	cl::Buffer cl_buffer; // OpenCL buffer for GPU operations

    Tensor() = default;

    // 建構子：根據形狀初始化 Tensor
    Tensor(const std::vector<int>& shape);

    // 建構子：從現有資料和形狀建立 Tensor (複製資料)
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    // 取得 Tensor 的總元素數量
    size_t size() const;

    // 根據索引取得元素 (const 版本)
    float get(const std::vector<int>& index) const;

    // 根據索引取得元素的引用 (可修改 Tensor 內容)
    float& get(const std::vector<int>& index);

    // 取得指向內部資料的指標 (謹慎使用)
    float* data_ptr();

    const float* data_ptr() const;

	void transfer_to_gpu();
	void transfer_to_cpu();

    // 打印 Tensor 的形狀和部分內容 (方便調試)
    void print(size_t limit = 10) const;

private:
    // 計算總元素數量
    size_t calculate_total_size() const;

    // 計算步長
    void calculate_strides();

    // 將多維索引轉換為一維線性索引 (使用步長)
    size_t get_linear_index(const std::vector<int>& index) const;
};

#endif // TENSOR_H