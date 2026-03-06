#include "model.hpp"
#include <stdexcept>
#include <iostream>

Model::Model() {
    is_compiled = false;
    loss_function = nullptr;
    optimizer = nullptr;
}

void Model::add_layer(Layer* layer) {
    if(is_compiled){
        throw std::runtime_error("Cannot add layer after model is compiled.");
    }
    layers.push_back(layer);
    std::cout << "Added layer: " << layer->get_name() << std::endl;
}

void Model::compile(Loss* loss_function_, Optimizer* optimizer_) {
    if(is_compiled){
        throw std::runtime_error("Model is already compiled.");
    }
    loss_function = loss_function_;
    optimizer = optimizer_;
    for (auto& layer : layers) {
        layer->pass_parameters(parameters, grad_parameters);
    }
    optimizer->initialize_momentum(parameters);
    is_compiled = true;
    std::cout << "Model compiled successfully with loss function and optimizer." << std::endl;
}

void Model::setIntermediateInputs(size_t max_batch_size) {
    for(auto t : intermediate_inputs) delete t;
    for(auto t : grad_intermediate_inputs) delete t;

    intermediate_inputs.clear();
    grad_intermediate_inputs.clear();
    for(auto& layer : layers) {
        layer->get_tensor(intermediate_inputs, max_batch_size);
        layer->get_tensor(grad_intermediate_inputs, max_batch_size);
        intermediate_inputs.back()->toGPU();
        grad_intermediate_inputs.back()->toGPU();
    }
}

std::vector<Tensor*> Model::auto_batching(std::vector<Tensor*>& inputs, size_t max_batch_size) {
    std::vector<Tensor*> batched_inputs;
    std::vector<size_t> batch_sizes;
    size_t total_sample = inputs.size();
    batch_sizes.clear();
    while(total_sample > 0){
        size_t current_batch_size = std::min<size_t>(max_batch_size, total_sample);
        batch_sizes.push_back(current_batch_size);
        total_sample -= current_batch_size;
    }
    size_t start_index = 0;
    std::vector<size_t> shape = inputs[0]->getShape();
    shape.insert(shape.begin(), 0);
    for(size_t bs : batch_sizes){
        shape[0] = bs;
        batched_inputs.emplace_back(new Tensor(shape));
        Tensor& batched_input = *batched_inputs.back();
        combine_tensors(std::vector<Tensor*>(inputs.begin() + start_index, inputs.begin() + start_index + bs), batched_input);
        batched_input.toGPU();
        start_index += bs;
    }

    return batched_inputs;
}




float Model::train(const Tensor& inputs, const Tensor& targets, size_t batch_size, float learning_rate) {
    if(!is_compiled){
        throw std::runtime_error("Model must be compiled before training.");
    }
    for(int i = 0; i < layers.size(); i++){
        if(i == 0){
            layers[i]->forward(inputs, *intermediate_inputs[i], batch_size);
        } else {
            layers[i]->forward(*intermediate_inputs[i-1], *intermediate_inputs[i], batch_size);
        }
    }
    
    float loss = loss_function->forward(*intermediate_inputs.back(), targets, batch_size);
    loss_function->backward(*intermediate_inputs.back(), targets, *grad_intermediate_inputs.back(), batch_size);
    

    Tensor dummy_grad_input(inputs.getShape());
    dummy_grad_input.toGPU();
    for(int i = layers.size() - 1; i >= 0; i--){
        if(i == 0){
            layers[i]->backward(*grad_intermediate_inputs[i], inputs, dummy_grad_input, batch_size);
        }
        else{
            layers[i]->backward(*grad_intermediate_inputs[i], *intermediate_inputs[i-1], *grad_intermediate_inputs[i-1], batch_size);
        }
        
    }
    optimizer->update(parameters, grad_parameters, learning_rate);
    return loss;
}

Tensor Model::predict(const Tensor& inputs) {
    if(!is_compiled){
        throw std::runtime_error("Model must be compiled before prediction.");
    }
    inputs.toGPU();
    for(int i = 0; i < layers.size(); i++){
        if(i == 0){
            layers[i]->predict(inputs, *intermediate_inputs[i]);
        } else {
            layers[i]->predict(*intermediate_inputs[i-1], *intermediate_inputs[i]);
        }
    }
    inputs.toCPU();
    return *intermediate_inputs.back();
}