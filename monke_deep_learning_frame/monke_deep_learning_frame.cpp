#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric> // std::accumulate
#include <string>
#include "Tensor.h"
#include <random>



using namespace std;

float activity_function(float x) {
	// Example activation function (ReLU_Leaky)
	return x > 0 ? x : 0.001*x;
}
float activity_function_derivative(float x) {
	// Derivative of the activation function (ReLU_Leaky)
	return x > 0 ? 1 : 0.001;
}

float random_normal_float(float mean, float stddev) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<float> dist(mean, stddev);
	return dist(gen);
}

//tensor is a class that represents a multi-dimensional array
//layer is a basis class
class layer {

public:
	// Constructor
	layer() {
		// Initialize the layer
	}
	// Destructor
	virtual ~layer() {
		// Clean up resources
	}
	// Forward pass
	virtual void forward(const Tensor& input,Tensor& output) = 0;
	// Backward pass
	virtual void backward(const Tensor& grad_output, Tensor& grad_input) = 0;
	// Update weights
	virtual void update(float learning_rate) = 0;
};
class dense : public layer {
public:
	// Constructor
	dense(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
		// Initialize weights and biases He initialization
		weights = vector<vector<float>>(output_size, vector<float>(input_size, 0.0f));
		biases.resize(output_size);
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				weights[i][j] = random_normal_float(0.0f, sqrt(2.0f / input_size));
			}
			biases[i] = random_normal_float(0.0f, sqrt(2.0f / input_size));
		}
		

	}
	// Destructor
	~dense() {
		// Clean up resources

	}
	// Forward pass
	void forward(const Tensor& input,Tensor& output) override {
		// Perform forward 
		// pass using weights and biases
		for (int i = 0; i < output_size; ++i) {
			float sum = 0;
			for (int j = 0; j < input_size; ++j) {
				sum += weights[i][j] * input.get({ j });
			}

			output.get({ i }) = sum + biases[i];
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, Tensor& grad_input) override {
		// Perform backward pass
	}
	// Update weights
	void update(float learning_rate) override {
		// Update weights and biases
	}
private:
	int input_size;
	int output_size;
	std::vector<vector<float>> weights; //weights are stored in a vector
	std::vector<float> biases; //biases are stored in a vector
	std::vector<float> grad_weights; //grad_weights are stored in a vector
	std::vector<float> grad_biases; //grad_biases are stored in a vector
};
//model is a class that contains layers

class model {

public:
	// Constructor
	model() {
		// Initialize the model
	}
	// Destructor
	~model() {
		// Clean up resources
	}
	// Add a layer to the model
	void addLayer(layer* l) {
		
	}
};

int main() {

    Tensor t1({ 2, 3, 4 });
    t1.print(t1.size());
    t1.get({ 1, 0, 2 }) = 5.0;
    std::cout << "t1.get({1, 0, 2}): " << t1.get({ 1, 0, 2 }) << std::endl;

    Tensor t2({ 5 }, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    t2.print();
    std::cout << "t2.get({3}): " << t2.get({ 3 }) << std::endl;

    try {
        t1.get({ 2, 0, 0 });
    }
    catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
