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
	size_t sizebyte; //for cl::Buffer
	cl::Buffer cl_buffer; // OpenCL buffer for GPU operations

    Tensor() = default;

    Tensor(const std::vector<int>& shape);

    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    Tensor(const Tensor& other);

    ~Tensor();

    size_t size() const;

    void copy_from(const Tensor& other);

	const std::vector<int>& get_shape() const;

    void set(const std::vector<int>& index, float value);

    float get(const std::vector<int>& index) const;


    float& get(const std::vector<int>& index);

	cl::Buffer get_buffer();

    cl::Buffer get_buffer() const;


    float* data_ptr();

    const float* data_ptr() const;

    void share_buffer_and_reshape(const Tensor& other, const std::vector<int>& new_shape);
	void transfer_to_gpu();
	void transfer_to_cpu();


    void print(size_t limit = 10) ;

private:

    size_t calculate_total_size() const;

    void calculate_strides();


    size_t get_linear_index(const std::vector<int>& index) const;
};

#endif // TENSOR_H