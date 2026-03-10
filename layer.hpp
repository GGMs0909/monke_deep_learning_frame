#pragma once
#include "opencl_runtime.hpp"
#include "tensor.hpp"
#include <vector>

class Layer {
public:
    virtual void predict(const Tensor& inputs, Tensor& outputs) = 0;
    virtual void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) = 0;
    virtual void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) = 0;
    virtual void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) = 0;
    virtual void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) = 0;
    virtual std::string get_name() const = 0;
};


class Scale : public Layer {
private:
    int size;
    float scale_factor;
    cl::make_kernel<cl::Buffer,cl::Buffer,int,float> forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,int,float> backward_kernel;
public:
    Scale(size_t size_, float scale_factor_);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};

class Dropout : public Layer {
private:
    int size;
    float p;
    cl::Buffer masks;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, float, unsigned int, int> forward_kernel;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, float, int> backward_kernel;
public:
    Dropout(size_t size_, float p_);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};




class Dense : public Layer {
private:
    int input_size;
    int output_size;
    Tensor weights;
    Tensor biases;
    Tensor grad_weights;
    Tensor grad_biases;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,cl::Buffer,int,int,int> forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,cl::Buffer,int,int,int> backward_wb_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int,int,int> backward_input_kernel;
public:
    Dense(size_t input_size_, size_t output_size_);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};

class Convolution : public Layer{
private:
    int input_channels;
    int input_size;
    int output_channels;
    int kernel_size;
    Tensor weights;
    Tensor biases;
    Tensor grad_weights;
    Tensor grad_biases;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int> forward_kernel;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int> backward_weight_kernel;
    cl::make_kernel<cl::Buffer, cl::Buffer, int, int, int> backward_bias_kernel;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int, int> backward_input_kernel;
public:
    Convolution(size_t input_channels_, size_t input_size_, size_t output_channels_, size_t kernel_size_);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};

class ReLU : public Layer{
private:
    int size;
    float slope;
    cl::make_kernel<cl::Buffer,cl::Buffer,int,float> relu_forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int,float> relu_backward_kernel;
public:
    ReLU(size_t size_, float slope_ = 0.0f);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};

class Softmax : public Layer{
private:
    int size;
    cl::Buffer outputs_backup;
    cl::make_kernel<cl::Buffer,cl::Buffer,int,int> forward_kernel;
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int,int> backward_kernel;
public:
    Softmax(size_t size_);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};

//batch_normalization

class BN : public Layer{
private:
    int size;
    float p;
    cl::Buffer means;
    cl::Buffer sqrtVars;
    cl::Buffer AvMeans;
    cl::Buffer AvSVs;
    Tensor gammas;
    Tensor grad_gammas;
    Tensor betas;
    Tensor grad_betas;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, float, int, int> forward_kernel_mean_sqrtvar;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> forward_kernel;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> backward_kernel_gb;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> backward_kernel_inputs;
public:
    BN(size_t size_, float p_ = 0.1);
    void predict(const Tensor& inputs, Tensor& outputs) override;
    void forward(const Tensor& inputs, Tensor& outputs, size_t batch_size) override;
    void backward(const Tensor& grad_outputs, const Tensor& inputs, Tensor& grad_inputs, size_t batch_size) override;
    void pass_parameters(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) override;
    void get_tensor(std::vector<Tensor*>& inputs, size_t batch_size) override;
    std::string get_name() const override;
};