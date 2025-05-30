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

    // �غc�l�G�ھڧΪ���l�� Tensor
    Tensor(const std::vector<int>& shape);

    // �غc�l�G�q�{����ƩM�Ϊ��إ� Tensor (�ƻs���)
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    // ���o Tensor ���`�����ƶq
    size_t size() const;

    // �ھگ��ި��o���� (const ����)
    float get(const std::vector<int>& index) const;

    // �ھگ��ި��o�������ޥ� (�i�ק� Tensor ���e)
    float& get(const std::vector<int>& index);

    // ���o���V������ƪ����� (�ԷV�ϥ�)
    float* data_ptr();

    const float* data_ptr() const;

	void transfer_to_gpu();
	void transfer_to_cpu();

    // ���L Tensor ���Ϊ��M�������e (��K�ո�)
    void print(size_t limit = 10) const;

private:
    // �p���`�����ƶq
    size_t calculate_total_size() const;

    // �p��B��
    void calculate_strides();

    // �N�h�������ഫ���@���u�ʯ��� (�ϥΨB��)
    size_t get_linear_index(const std::vector<int>& index) const;
};

#endif // TENSOR_H