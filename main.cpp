// main.cpp
#include <iostream>
#include <vector>
#include "opencl_runtime.hpp"
#include "tensor.hpp"
#include "model.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"


using namespace std;
int main() {
    opencl_runtime::getInstance().initialize();
    Model model;
    model.add_layer(new Dense(2, 4));
    model.add_layer(new ReLU(4));
    model.add_layer(new Dense(4, 1));
    model.compile(new MSE(1), new Adam(0.9, 0.999, 1e-8));
    cout << "-------------------" << endl;
    //want to learn the function f(x) = 2x1^2 + 3x1x2 + 4x2^2
    vector<Tensor> inputs = {Tensor({2}), Tensor({2}), Tensor({2}), Tensor({2})};
    vector<Tensor> targets = {Tensor({1}), Tensor({1}), Tensor({1}), Tensor({1})};
    for(int i = 0; i < 4; i++){
        inputs[i].set({0}, i+1);
        inputs[i].set({1}, i+2);
        float x1 = inputs[i].get({0});
        float x2 = inputs[i].get({1});
        targets[i].set({0}, 2*x1*x1 + 3*x1*x2 + 4*x2*x2);
    }
    vector<Tensor*> input_ptrs, target_ptrs;
    vector<Tensor*> batched_inputs, batched_targets;
    for(int i = 0; i < 4; i++){
        input_ptrs.push_back(&inputs[i]);
        target_ptrs.push_back(&targets[i]);
    }
    vector<size_t> batch_sizes;
    model.auto_batching(input_ptrs, batched_inputs, 2, batch_sizes);
    model.auto_batching(target_ptrs, batched_targets, 2, batch_sizes);
    model.setIntermediateInputs(2);
    cout << "--------------------" << endl;
    cout << "Starting training..." << endl;
    float total_loss = 0;
    for(int epoch = 0; epoch < 10000; epoch++){
        total_loss = 0;
        for(int i = 0; i < batched_inputs.size(); i++){
            total_loss += model.train(*batched_inputs[i], *batched_targets[i], batch_sizes[i], 0.001f);
        }
        if(epoch % 100 == 0){
            cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << endl;
        }
    }
    //predict
    model.setIntermediateInputs(1);
    Tensor test_input({2});
    test_input.set({0}, 5);
    test_input.set({1}, 6);
    Tensor test_output({1});
    model.predict(test_input, test_output);
    test_output.toCPU();
    test_output.print();
    cout << "Test input: [5, 6], Predicted output: " << test_output.get({0,0}) << ", Expected output: " << 2*5*5 + 3*5*6 + 4*6*6 << endl;

    return 0;
}