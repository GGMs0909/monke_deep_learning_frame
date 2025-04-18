#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric> // std::accumulate
#include <string>
#include "Tensor.h"
#include <random>



using namespace std;

double activity_function(double x) {
	// Example activation function (ReLU_Leaky)
	return x > 0 ? x : 0.01*x;
}
double activity_function_derivative(double x) {
	// Derivative of the activation function (ReLU_Leaky)
	return x > 0 ? 1 : 0.01;
}

double random_normal_double(double mean, double stddev) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<double> dist(mean, stddev);
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
	virtual void update(double learning_rate) = 0;
	// Get input size
	int get_input_size() const {
		return input_size;
	}
private:
	// Private members
	int input_size; //input size of the layer
	int output_size; //output size of the layer
	vector<Tensor> weights; //weights are stored in a vector
	vector<double> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<double> grad_biases; //grad_biases are stored in a vector
};
class dense : public layer {
public:
	// Constructor
	dense(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
		// Initialize weights and biases He initialization
		weights = vector<Tensor>(output_size, Tensor({ input_size }));
		biases = vector<double>(output_size,0);
		grad_weights = weights;
		grad_biases = biases;
		// Initialize weights and biases using He initialization
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				weights[i].get({j}) = random_normal_double(0.0f, sqrt(2.0f / input_size));
			}
			biases[i] = random_normal_double(0.0f, sqrt(2.0f / input_size));
		}
		

	}
	// Destructor
	~dense() {
		// Clean up resources
		// No dynamic memory allocation, so nothing to clean up

	}
	// Forward pass
	void forward(const Tensor& input,Tensor& output) override {
		// Perform forward 
		// pass using weights and biases
		for (int i = 0; i < output_size; ++i) {
			double sum = 0;
			for (int j = 0; j < input_size; ++j) {
				sum += weights[i].get({j}) * input.get({j});
			}

			output.get({ i }) = sum + biases[i];
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, Tensor& grad_input) override {
		// Perform backward pass
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				grad_weights[i].get({j}) += grad_output.get({i}) * grad_input.get({j});
			}
			grad_biases[i] += grad_output.get({ i });
		}
		for (int i = 0; i < input_size; ++i) {
			double sum = 0;
			for (int j = 0; j < output_size; ++j) {
				sum += grad_output.get({ j }) * weights[j].get({i});
			}
			grad_input.get({ i }) += sum;
		}

	}
	// Update weights
	void update(double learning_rate) override {
		// Update weights and biases
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				weights[i].get({j}) -= learning_rate * grad_weights[i].get({j});
			}
			biases[i] -= learning_rate * grad_biases[i];
		}
	}
private:
	
	int input_size;
	int output_size;
	vector<Tensor> weights; //weights are stored in a vector
	vector<double> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<double> grad_biases; //grad_biases are stored in a vector
	
};
class convolution : public layer {
public:
	// Constructor
	convolution(int input_channels, int input_size, int output_channels, int kernel_size)
		: input_channels(input_channels), input_size(input_size), output_channels(output_channels), kernel_size(kernel_size) {
		// Initialize weights and biases
		weights = vector<Tensor>(output_channels, Tensor({ input_channels, kernel_size, kernel_size }));
		biases = vector<double>(output_channels,0);
		grad_weights = weights;
		grad_biases = biases;
		// Initialize weights and biases using He initialization
		
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < kernel_size; ++k) {
					for (int l = 0; l < kernel_size; ++l) {
						weights[i].get({ j,k,l }) = random_normal_double(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));
					}
				}
			}
			biases[i] = random_normal_double(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));
		}

	}
	// Destructor
	~convolution() {
		// Clean up resources
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override{
		// Perform forward pass using weights and biases
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < input_size - kernel_size + 1; ++k) {
					for (int l = 0; l < input_size - kernel_size + 1; ++l) {
						double sum = 0;
						for (int m = 0; m < kernel_size; ++m) {
							for (int n = 0; n < kernel_size; ++n) {
								sum += input.get({ j, k + m, l + n }) * weights[i].get({ j, m, n });
							}
						}
						output.get({ i, k, l }) = activity_function(sum + biases[i]);
					}
				}
			}
		}
		
	}
	// Backward pass
	void backward(const Tensor& grad_output, Tensor& grad_input) override{
		// Perform backward pass
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < input_size - kernel_size + 1; ++k) {
					for (int l = 0; l < input_size - kernel_size + 1; ++l) {
						double sum = 0;
						for (int m = 0; m < kernel_size; ++m) {
							for (int n = 0; n < kernel_size; ++n) {
								sum += grad_output.get({ i, k, l }) * weights[i].get({j,m,n});
								grad_weights[i].get({j,m,n}) += grad_output.get({i, k, l}) * grad_input.get({j, k + m, l + n});
							}
						}
						grad_input.get({ j, k, l }) = sum;
					}
				}
			}
		}
		
	}
	// Update weights
	void update(double learning_rate) override{
		// Update weights and biases
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < kernel_size; ++k) {
					for (int l = 0; l < kernel_size; ++l) {
						weights[i].get({ j, k, l }) -= learning_rate * grad_weights[i].get({ j, k, l });

					}
				}
			}
			biases[i] -= learning_rate * grad_biases[i];
		}
	}
private:
	int input_channels;
	int input_size;
	int output_channels;

	int kernel_size;
	vector<Tensor> weights; //weights are stored in a vector
	vector<double> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<double> grad_biases; //grad_biases are stored in a vector
	
};
class pooling : public layer {
public:
	// Constructor
	pooling(int input_channels, int input_size, int pool_size) : input_channels(input_channels), input_size(input_size), pool_size(pool_size) {
		// Initialize weights and biases
		weights = vector<Tensor>(input_channels, Tensor({ pool_size, pool_size }));
		biases = vector<double>(input_channels, 0);
		grad_weights = weights;
		grad_biases = biases;
	}
	// Destructor
	~pooling() {
		// Clean up resources
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		// Perform forward pass using weights and biases
		for (int i = 0; i < input_channels; ++i) {
			for (int j = 0; j < input_size - pool_size + 1; ++j) {
				for (int k = 0; k < input_size - pool_size + 1; ++k) {
					double max_val = -std::numeric_limits<double>::max();
					for (int m = 0; m < pool_size; ++m) {
						for (int n = 0; n < pool_size; ++n) {
							max_val = std::max(max_val, input.get({ i, j + m, k + n }));
						}
					}
					output.get({ i, j / pool_size, k / pool_size }) = max_val;
				}
			}
		}

	}

	void backward(const Tensor& grad_output, Tensor& grad_input) override {

	}

	void update(double learning_rate) override {

	}
	
private:
	int input_channels;
	int input_size;
	int pool_size;
	vector<Tensor> weights; //weights are stored in a vector
	vector<double> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<double> grad_biases; //grad_biases are stored in a vector
};
//model is a class that contains layers

class model {

public:
	// Constructor
	model(vector<layer*> layers) : layers(layers) {
		// Initialize the model
		// Initialize inputs


	}
	// Destructor
	~model() {
		// Clean up resources
	}
	

private:
	vector<layer*> layers; //layers are stored in a vector
	vector<Tensor> inputs; //inputs are stored in a vector
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
