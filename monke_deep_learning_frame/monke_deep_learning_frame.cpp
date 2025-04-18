#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric> // std::accumulate
#include <string>
#include "Tensor.h"
#include <random>



using namespace std;



static double random_normal_double(double mean, double stddev) {
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
	virtual void backward(const Tensor& grad_output, const Tensor& output, Tensor& grad_input) = 0;
	// Update weights
	virtual void update(double learning_rate) = 0;

private:
	// Private members
	
};
class relu_1D : public layer {
public:
	relu_1D(int input_size,int output_size) : input_size(input_size), output_size(output_size) {
		// Initialize the ReLU layer
	}
	// Destructor
	~relu_1D() {
		// Clean up resources
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		// Perform forward pass using ReLU activation function
		for (int i = 0; i < input_size; ++i) {
			output.get({ i }) = std::max(0.0, input.get({ i }));
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		// Perform backward pass using ReLU activation function
		for (int i = 0; i < input_size; ++i) {
			if (input.get({ i }) > 0) {
				grad_input.get({ i }) = grad_output.get({ i });
			}
			else {
				grad_input.get({ i }) = 0;
			}
		}
	}
private:
	int input_size;
	int output_size;

};
class relu_3D : public layer {
public:
	relu_3D(int input_channels, int input_size, int output_channels) : input_channels(input_channels), input_size(input_size), output_channels(output_channels) {
		// Initialize the ReLU layer
	}
	// Destructor
	~relu_3D() {
		// Clean up resources
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		// Perform forward pass using ReLU activation function
		for (int h = 0; h < output_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					output.get({ h, r, c }) = std::max(0.0, input.get({ h, r, c }));
				}
			}
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		// Perform backward pass using ReLU activation function
		for (int h = 0; h < output_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					if (input.get({ h, r, c }) > 0) {
						grad_input.get({ h, r, c }) = grad_output.get({ h, r, c });
					}
					else {
						grad_input.get({ h, r, c }) = 0;
					}
				}
			}
		}

	}
private:
	int input_channels;
	int input_size;
	int output_channels;
	// Private members
	// Add any additional private members if needed
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
	void backward(const Tensor& grad_output,const Tensor& input, Tensor& grad_input) override {
		// Perform backward pass
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				grad_weights[i].get({j}) += grad_output.get({i}) * input.get({j});
			}
			grad_biases[i] += grad_output.get({ i });
		}
		for (int i = 0; i < input_size; ++i) {
			double sum = 0;
			for (int j = 0; j < output_size; ++j) {
				sum += grad_output.get({ j }) * weights[j].get({ i });
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
		for (int h = 0; h < output_channels; h++) {
			for (int r = 0; r < input_size - kernel_size + 1;r++) {
				for (int c = 0; c < input_size - kernel_size + 1;c++) {
					double sum = 0;
					for (int i = 0; i < input_channels; i++) {
						for (int j = 0; j < kernel_size; j++) {
							for (int k = 0; k < kernel_size; k++) {
								sum += input.get({ i, r + j, c + k }) * weights[h].get({ i, j, k });
							}
						}
					}
					output.get({ h, r, c }) = sum + biases[h];
				}
			}
		}
		
	}
	// Backward pass
	void backward(const Tensor& grad_output,const Tensor& input, Tensor& grad_input) override{
		// Perform backward pass
		for (int h = 0; h < output_channels; ++h) {
			for (int i = 0; i < input_channels; ++i) {
				for (int j = 0; j < kernel_size; ++j) {
					for (int k = 0; k < kernel_size; ++k) {
						double sum = 0;
						for (int r = 0; r < input_size - kernel_size + 1; ++r) {
							for (int c = 0; c < input_size - kernel_size + 1; ++c) {
								sum += grad_output.get({ h, r, c }) * input.get({ i, r + j, c + k });
							}
						}
						grad_weights[h].get({ i, j, k }) += sum;
					}
				}
			}
			double sum = 0;
			for (int r = 0; r < input_size - kernel_size + 1;++r) {
				for (int c = 0; c < input_size - kernel_size + 1;++c) {
					sum += grad_output.get({ h, r, c });
				}
			}
			grad_biases[h] += sum;
		}
		for (int i = 0; i < input_channels; ++i) {
			for (int j = 0; j < input_size; ++j) {
				for (int k = 0; k < input_size; ++k) {
					double sum = 0;
					for (int h = 0; h < output_channels; ++h) {
						for (int r = 0; r < kernel_size; ++r) {
							for (int c = 0; c < kernel_size; ++c) {
								if (j - r >= 0 && j - r < input_size && k - c >= 0 && k - c < input_size) {
									sum += grad_output.get({ h, j - r, k - c }) * weights[h].get({ i, r, c });
								}
							}
						}
					}
					grad_input.get({ i, j, k }) += sum;
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
	pooling(int input_channels, int input_size, int pool_size)
		: input_channels(input_channels), input_size(input_size), pool_size(pool_size) {
		
	}
	// Destructor
	~pooling() override {}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		// Store the indices of the maximum values for backward pass
		count = vector<vector<vector<int>>>(input_channels, vector<vector<int>>(input_size - pool_size + 1, vector<int>(input_size - pool_size + 1, 0)));

		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < input_size - pool_size + 1; ++r) {
				for (int c = 0; c < input_size - pool_size + 1; ++c) {
					double max_val = input.get({ h, r, c });
					for (int i = 0; i < pool_size; ++i) {
						for (int j = 0; j < pool_size; ++j) {
							if (input.get({ h, r + i, c + j }) > max_val) {
								max_val = input.get({ h, r + i, c + j });
							}
						}
					}
					output.get({ h, r, c}) = max_val;
					for (int i = 0; i < pool_size; ++i) {
						for (int j = 0; j < pool_size; ++j) {
							if (input.get({ h, r + i, c + j }) == max_val) {
								count[h][r][c]++;
							}
						}
					}
				}
			}
		}
	}

	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					double sum = 0;
					for (int i = 0; i < pool_size; ++i) {
						for (int j = 0; j < pool_size; ++j) {
							if (r - i >= 0 && r - i < input_size && c - j >= 0 && c - j < input_size) {
								sum += ((grad_output.get({ h, r - i, c - j }) == input.get({h,r,c})) / count[h][r - i][c - j]);
							}
						}
					}
					grad_input.get({ h, r, c }) += sum;
				}
			}
		}
	}

	void update(double learning_rate) override {
		// do nothing
	}

private:
	int input_channels;
	int input_size;
	int pool_size;
	vector<vector<vector<int>>> count; // Store counts of max values
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
