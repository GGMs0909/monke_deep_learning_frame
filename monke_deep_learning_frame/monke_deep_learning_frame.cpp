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
	virtual string get_name() = 0;
	// Initialize Tensor
	virtual void Get_Tensor(Tensor& output) = 0;
	// Forward pass
	virtual void forward(const Tensor& input,Tensor& output) = 0;
	// Backward pass
	virtual void backward(const Tensor& grad_output, const Tensor& output, Tensor& grad_input) = 0;
	// Update weights
	virtual void update(double learning_rate) = 0;
	// Reset gradients
	virtual void reset_gradients() = 0;
	// get layer name
	

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
	string get_name() override {
		return "relu_1D";
	}
	//Initialize Tensor
	void Get_Tensor(Tensor& output) override {
		output = Tensor({output_size});
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
	// Update weights
	void update(double learning_rate) override {
		// do nothing
	}
	// Reset gradients
	void reset_gradients() override {
		// do nothing
	}
private:
	int input_size;
	int output_size;

};
class relu_3D : public layer {
public:
	relu_3D(int input_channels, int input_size) : input_channels(input_channels), input_size(input_size) {
		// Initialize the ReLU layer
	}
	// Destructor
	~relu_3D() {
		// Clean up resources
	}
	string get_name() override {
		return "relu_3D";
	}
	void Get_Tensor(Tensor& output) override {
		output = Tensor({ input_channels,input_size,input_size });
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		// Perform forward pass using ReLU activation function
		int output_channels = input_channels;
		for (int h = 0; h < output_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					//cout << h << " " << r << " " << c << endl;
					output.get({ h, r, c }) = std::max(0.0, input.get({ h, r, c }));
				}
			}
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		// Perform backward pass using ReLU activation function
		int output_channels = input_channels;
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
	// Update weights
	void update(double learning_rate) override {
		// do nothing
	}
	// Reset gradients
	void reset_gradients() override {
		// do nothing
	}
private:
	int input_channels;
	int input_size;

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
	string get_name() override {
		return "dense";
	}
	void Get_Tensor(Tensor& output) override {
		output = Tensor({output_size});
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
	// Reset gradients
	void reset_gradients() override {
		// Reset gradients
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				grad_weights[i].get({ j }) = 0;
			}
			grad_biases[i] = 0;
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
	string get_name() override {
		return "convolution";
	}
	void Get_Tensor(Tensor& output) override {
		output = Tensor({ output_channels,input_size-kernel_size+1,input_size-kernel_size+1 });
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
								if (j - r >= 0 && j - r < input_size-kernel_size+1 && k - c >= 0 && k - c < input_size-kernel_size+1) {
									//cout << h << " " << j - r << " " << k - c << endl;
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
	// Reset gradients
	void reset_gradients() override {
		// Reset gradients
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < kernel_size; ++k) {
					for (int l = 0; l < kernel_size; ++l) {
						grad_weights[i].get({ j, k, l }) = 0;
					}
				}
			}
			grad_biases[i] = 0;
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
		// Output size for pooling with stride = pool_size (common),
		// but your forward uses stride 1, so output size is input_size - pool_size + 1
		int output_dim = input_size - pool_size + 1;
		if (output_dim <= 0) {
			throw std::invalid_argument("Invalid input_size or pool_size for pooling layer.");
		}
		// Initialize max_indices storage based on output dimensions
		max_indices = vector<vector<vector<pair<int, int>>>>(input_channels,
			vector<vector<pair<int, int>>>(output_dim,
				vector<pair<int, int>>(output_dim)));
	}

	// Destructor
	~pooling() override = default; // Using default destructor

	string get_name() override {
		return "pooling";
	}

	void Get_Tensor(Tensor& output) override {
		int output_dim = input_size - pool_size + 1;
		if (output_dim <= 0) {
			throw std::invalid_argument("Invalid input_size or pool_size for pooling layer.");
		}
		output = Tensor({ input_channels, output_dim, output_dim });
	}

	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		int output_dim = input_size - pool_size + 1;

		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < output_dim; ++r) {
				for (int c = 0; c < output_dim; ++c) {
					double max_val = input.get({ h, r, c }); // Initialize with the first element in the window
					int max_i = 0; // Relative row index of the max value in the window
					int max_j = 0; // Relative col index of the max value in the window

					for (int i = 0; i < pool_size; ++i) {
						for (int j = 0; j < pool_size; ++j) {
							if (input.get({ h, r + i, c + j }) > max_val) {
								max_val = input.get({ h, r + i, c + j });
								max_i = i;
								max_j = j;
							}
						}
					}
					output.get({ h, r, c }) = max_val;
					max_indices[h][r][c] = { max_i, max_j }; // Store the relative index of the max
				}
			}
		}
	}

	// Backward pass (Modified for standard max pooling)
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		// grad_input needs to be initialized to zero before the backward pass starts for a batch
		int output_dim = input_size - pool_size + 1;

		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < output_dim; ++r) {
				for (int c = 0; c < output_dim; ++c) {
					// Get the stored relative index of the max value for this output gradient
					pair<int, int> max_idx = max_indices[h][r][c];
					int max_i = max_idx.first;
					int max_j = max_idx.second;

					// Add the output gradient to the location of the max value in the input gradient
					// The location in input is (h, r + max_i, c + max_j)
					grad_input.get({ h, r + max_i, c + max_j }) += grad_output.get({ h, r, c });
				}
			}
		}
	}

	void update(double learning_rate) override {
		// Pooling layers have no learnable parameters, so nothing to update
	}

	void reset_gradients() override {
		// Pooling layers have no learnable parameters or gradients to reset
	}

private:
	int input_channels;
	int input_size;
	int pool_size;
	// Store the relative (i, j) index of the maximum value in the pool_size x pool_size window for each output element
	vector<vector<vector<pair<int, int>>>> max_indices;
};
//flatten_3D 3D -> 1D
class flatten_3D : public layer {
public:
	// Constructor
	flatten_3D(int input_channels, int input_size) : input_channels(input_channels), input_size(input_size) {
		// Initialize the flatten layer
	}
	// Destructor
	~flatten_3D() {
		// Clean up resources
	}
	string get_name() override {
		return "flatten_3D";
	}
	void Get_Tensor(Tensor& output) override {
		output = Tensor({ input_channels * input_size * input_size });
	}
	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					output.get({ h * input_size * input_size + r * input_size + c }) = input.get({ h, r, c });
					
				}
			}
		}
	}
	// Backward pass
	void backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) override {
		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					grad_input.get({ h, r, c }) += grad_output.get({ h * input_size * input_size + r * input_size + c });
				}
			}
		}
	}
	// Update weights
	void update(double learning_rate) override {
		// do nothing
	}
	// Reset gradients
	void reset_gradients() override {
		// do nothing
	}
private:
	int input_channels;
	int input_size;

};
//model is a class that contains layers

class model {

public:
	// Constructor
	model(vector<layer*> layers) : layers(layers) {
		// Initialize the model
		// Initialize inputs
		inputs = vector<Tensor>(layers.size()+1);
		grad_inputs = vector<Tensor>(layers.size() + 1);
		for(int i = 0; i < layers.size(); i++){
			layers[i]->Get_Tensor(inputs[i+1]);
			grad_inputs[i+1] = inputs[i+1];
			cout << layers[i]->get_name() << endl;
			inputs[i+1].print();
		}

	}
	// Destructor
	~model() {
		// Clean up resources
	}
	Tensor forward(Tensor input) {
		cout << "---------forward---------" << endl;
		inputs[0] = input;
		for (int i = 0; i < layers.size(); i++) {
			cout << layers[i]->get_name() << endl;
			layers[i]->forward(inputs[i], inputs[i+1]);
		}
		cout << "---------forward end----------" << endl;
		return inputs.back();
	}

	void backward(Tensor grad_output) {
		grad_inputs.back() = grad_output;
		grad_inputs[0] = inputs[0];//just for a shape
		for (int i = layers.size() - 1; i >= 0; i--) {
			cout << layers[i]->get_name() << endl;
			layers[i]->backward(grad_inputs[i+1], inputs[i], grad_inputs[i]);
			
		}
	}
	void update(double learning_rate) {
		for (int i = 1; i < layers.size(); i++) {
			layers[i]->update(learning_rate);
		}
	}
	void reset_gradients() {
		for (int i = 1; i < layers.size(); i++) {
			layers[i]->reset_gradients();
		}
	}

private:
	vector<layer*> layers; //layers are stored in a vector
	vector<Tensor> inputs; //inputs are stored in a vector ||  inputs[0] = userinput
	vector<Tensor> grad_inputs;
};

int main() {
	//Example usage of the Tensor class
    Tensor t1({ 2, 3, 4 });
    t1.print(t1.size());
    t1.get({ 1, 0, 2 }) = 5.0;
    std::cout << "t1.get({1, 0, 2}): " << t1.get({ 1, 0, 2 }) << std::endl;

    Tensor t2({ 5 }, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    t2.print();
    std::cout << "t2.get({3}): " << t2.get({ 3 }) << std::endl;


	//Example usage of the model class
	vector<layer*> layers;
	layers.push_back(new convolution(10, 10, 20, 3));
	layers.push_back(new relu_3D(20, 8));
	layers.push_back(new pooling(20, 8, 2));
	layers.push_back(new flatten_3D(20, 7));
	layers.push_back(new dense(20*7*7, 5));
	model m(layers);
	// Create input tensor
	Tensor input({ 10, 10, 10 });
	// Initialize input tensor with random values
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			for (int k = 0; k < 10; ++k) {
				input.get({ i, j, k }) = random_normal_double(0.0f, 1.0f);
			}
		}
	}

	cout << "gaga" << endl;
	
	Tensor output = m.forward(input);
	Tensor grad_output({ 5 });
	grad_output.get({ 0 }) = 1.0;
	grad_output.get({ 1 }) = 2.0;
	grad_output.get({ 2 }) = 3.0;
	grad_output.get({ 3 }) = 4.0;
	grad_output.get({ 4 }) = 5.0;
	// Perform backward pass
	m.backward(grad_output);
	m.update(0.01);
	m.reset_gradients();
	// Clean up
	for (auto layer : layers) {
		delete layer;
	}

    return 0;
}
