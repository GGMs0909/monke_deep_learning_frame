#pragma once
#include <string>
#include <vector>
#include "tensor.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "opencl_runtime.hpp"

class Model {
private:
    bool is_compiled;
    std::vector<Layer*> layers;
    Loss* loss_function;
    Optimizer* optimizer;
    std::vector<Tensor*> parameters;
    std::vector<Tensor*> grad_parameters;
    std::vector<Tensor*> intermediate_inputs;
    std::vector<Tensor*> grad_intermediate_inputs;
public:
    Model();
    void add_layer(Layer* layer);
    void compile(Loss* loss_function_, Optimizer* optimizer_);
    std::vector<Tensor*> auto_batching(std::vector<Tensor*>& inputs, size_t max_batch_size);
    void setIntermediateInputs(size_t max_batch_size);
    float train(const Tensor& inputs, const Tensor& targets, size_t batch_size, float learning_rate);
    Tensor predict(const Tensor& inputs);
};