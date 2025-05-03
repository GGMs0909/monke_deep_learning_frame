#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric> // std::accumulate
#include <string>
#include "Tensor.h"
#include <random>
#include <omp.h>



using namespace std;



static float random_normal_float(float mean, float stddev) {
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
	virtual string get_name() = 0;
	// Initialize Tensor
	virtual void Get_Tensor(Tensor& output) = 0;
	// Forward pass
	virtual void forward(const Tensor& input,Tensor& output) = 0;
	// Backward pass
	virtual void backward(const Tensor& grad_output, const Tensor& output, Tensor& grad_input) = 0;
	// Update weights
	virtual void update(float learning_rate) = 0;
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
			output.get({ i }) = max(0.0f, input.get({ i }));
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
	void update(float learning_rate) override {
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
#pragma omp for collapse(3)
		for (int h = 0; h < output_channels; ++h) {
			for (int r = 0; r < input_size; ++r) {
				for (int c = 0; c < input_size; ++c) {
					//cout << h << " " << r << " " << c << endl;
					output.get({ h, r, c }) = max(0.0f, input.get({ h, r, c }));
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
	void update(float learning_rate) override {
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
		biases = vector<float>(output_size,0);
		grad_weights = weights;
		grad_biases = biases;
		// Initialize weights and biases using He initialization
		for (int i = 0; i < output_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				weights[i].get({j}) = random_normal_float(0.0f, sqrt(2.0f / input_size));
			}
			biases[i] = random_normal_float(0.0f, sqrt(2.0f / input_size));
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
			float sum = 0;
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
			float sum = 0;
			for (int j = 0; j < output_size; ++j) {
				sum += grad_output.get({ j }) * weights[j].get({ i });
			}
			grad_input.get({ i }) += sum;
		}

	}
	// Update weights
	void update(float learning_rate) override {
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
	vector<float> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<float> grad_biases; //grad_biases are stored in a vector
	
};
class convolution : public layer {
public:
	// Constructor
	convolution(int input_channels, int input_size, int output_channels, int kernel_size)
		: input_channels(input_channels), input_size(input_size), output_channels(output_channels), kernel_size(kernel_size) {
		// Initialize weights and biases
		weights = vector<Tensor>(output_channels, Tensor({ input_channels, kernel_size, kernel_size }));
		biases = vector<float>(output_channels,0);
		grad_weights = weights;
		grad_biases = biases;
		// Initialize weights and biases using He initialization
		
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_channels; ++j) {
				for (int k = 0; k < kernel_size; ++k) {
					for (int l = 0; l < kernel_size; ++l) {
						weights[i].get({ j,k,l }) = random_normal_float(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));
					}
				}
			}
			biases[i] = random_normal_float(0.0f, sqrt(2.0f / (input_channels * kernel_size * kernel_size)));
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
		vector<vector<vector<float>>> input_vector(input_channels, vector<vector<float>>(input_size, vector<float>(input_size)));
		vector<vector<vector<float>>> output_vector(output_channels, vector<vector<float>>(input_size - kernel_size + 1, vector<float>(input_size - kernel_size + 1)));
		
		for (int i = 0; i < input_channels; ++i) {
			for (int j = 0; j < input_size; ++j) {
				for (int k = 0; k < input_size; ++k) {
					input_vector[i][j][k] = input.get({ i,j,k });
				}
			}
		}
		double im2col_time = 0;
		double conv_time = 0;
		for (int h = 0; h < output_channels; ++h) {
			vector<float>& weights_vector = weights[h].data;
			#pragma omp parallel
			{
				vector<float> thread_im2col(input_channels * kernel_size * kernel_size);
				#pragma omp for collapse(2)
				for (int r = 0; r < input_size - kernel_size + 1; ++r) {
					for (int c = 0; c < input_size - kernel_size + 1; ++c) {
						//im2col
						double start_time = omp_get_wtime();
						
						for (int i = 0; i < input_channels; ++i) {
							for (int j = 0; j < kernel_size; ++j) {
								for (int k = 0; k < kernel_size; ++k) {
									thread_im2col[i * kernel_size * kernel_size + j * kernel_size + k] = input_vector[i][r + j][c + k];
								}
							}
						}
						double end_time = omp_get_wtime();
						im2col_time += end_time - start_time;
						
						//convolution
						start_time = omp_get_wtime();
						float sum = 0;
						//using simd
						#pragma omp simd
						for (int i = 0; i < input_channels * kernel_size * kernel_size; ++i) {
							sum += thread_im2col[i] * weights_vector[i];
						}
						output_vector[h][r][c] = sum + biases[h];
						end_time = omp_get_wtime();
						conv_time += end_time - start_time;
					}
				}
					

			}
		}


		// Copy output_vector to output tensor
		for (int i = 0; i < output_channels; ++i) {
			for (int j = 0; j < input_size - kernel_size + 1; ++j) {
				for (int k = 0; k < input_size - kernel_size + 1; ++k) {
					output.get({ i,j,k }) = output_vector[i][j][k];
				}
			}
		}

		cout << "im2col time: " << im2col_time << endl;
		cout << "conv time: " << conv_time << endl;
		//cin.get();

		
	}
	// Backward pass
	void backward(const Tensor& grad_output,const Tensor& input, Tensor& grad_input) override{
		// Perform backward pass
		for (int h = 0; h < output_channels; ++h) {
			for (int i = 0; i < input_channels; ++i) {
				for (int j = 0; j < kernel_size; ++j) {
					for (int k = 0; k < kernel_size; ++k) {
						float sum = 0;
						for (int r = 0; r < input_size - kernel_size + 1; ++r) {
							for (int c = 0; c < input_size - kernel_size + 1; ++c) {
								sum += grad_output.get({ h, r, c }) * input.get({ i, r + j, c + k });
							}
						}
						grad_weights[h].get({ i, j, k }) += sum;
					}
				}
			}
			float sum = 0;
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
					float sum = 0;
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
					//cout << i << " " << j << " " << k << '\n';
				}
			}
		}
		
	}
	// Update weights
	void update(float learning_rate) override{
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
	vector<float> biases; //biases are stored in a vector
	vector<Tensor> grad_weights; //grad_weights are stored in a vector
	vector<float> grad_biases; //grad_biases are stored in a vector
	
};
class pooling : public layer {
public:
	// Constructor
	pooling(int input_channels, int input_size, int pool_size)
		: input_channels(input_channels), input_size(input_size), pool_size(pool_size) {
		// Output size for pooling with stride = pool_size (common),
		// but your forward uses stride 1, so output size is input_size - pool_size + 1
		int output_dim = input_size / pool_size;
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
		int output_dim = input_size / pool_size;
		if (output_dim <= 0) {
			throw std::invalid_argument("Invalid input_size or pool_size for pooling layer.");
		}
		output = Tensor({ input_channels, output_dim, output_dim });
	}

	// Forward pass
	void forward(const Tensor& input, Tensor& output) override {
		//int stride = pool_size;
		int output_dim = input_size / pool_size;
#pragma omp for collapse(3)
		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < output_dim; ++r) {
				for (int c = 0; c < output_dim; ++c) {
					float max_val = input.get({ h, r*pool_size, c*pool_size }); // Initialize with the first element in the window
					int max_i = 0; // Relative row index of the max value in the window
					int max_j = 0; // Relative col index of the max value in the window

					for (int i = 0; i < pool_size; ++i) {
						for (int j = 0; j < pool_size; ++j) {
							if (input.get({ h, r*pool_size + i, c*pool_size + j }) > max_val) {
								max_val = input.get({ h, r*pool_size + i, c*pool_size + j});
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
		int output_dim = input_size / pool_size;

		for (int h = 0; h < input_channels; ++h) {
			for (int r = 0; r < output_dim; ++r) {
				for (int c = 0; c < output_dim; ++c) {
					// Get the stored relative index of the max value for this output gradient
					pair<int, int> max_idx = max_indices[h][r][c];
					int max_i = max_idx.first;
					int max_j = max_idx.second;

					// Add the output gradient to the location of the max value in the input gradient
					// The location in input is (h, r + max_i, c + max_j)
					grad_input.get({ h, r*pool_size + max_i, c*pool_size + max_j}) += grad_output.get({h, r, c});
				}
			}
		}
	}

	void update(float learning_rate) override {
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
	void update(float learning_rate) override {
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
	model()  {
		// Initialize the model
		// Initialize inputs
		

	}
	// Destructor
	~model() {
		// Clean up resources
		for (int i = 0; i < layers.size(); i++) {
			delete layers[i];
		}
	}
	void add_layer(layer* l) {
		// Add a layer to the model
		cout << "add layer " << l->get_name() << endl;
		layers.push_back(l);
	}
	void compile_model() {
		inputs = vector<Tensor>(layers.size() + 1);
		grad_inputs = vector<Tensor>(layers.size() + 1);
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->Get_Tensor(inputs[i + 1]);
			grad_inputs[i + 1] = inputs[i + 1];
			cout << layers[i]->get_name() << endl;
			inputs[i + 1].print();
		}
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
	void update(float learning_rate) {
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
#pragma omp parallel
	{
		// omp_get_thread_num() 獲取當前執行緒的 ID (通常從 0 開始)
		// omp_get_num_threads() 獲取當前平行區域中的總執行緒數量
		int thread_id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		string s = "Hello from thread " + to_string(thread_id) + " out of " + to_string(num_threads) + '\n';
		std::cout << s;

		// 注意：平行區域內的輸出順序是不確定的
	}
	cin.get();

	// 在平行區域結束後，只有主執行緒會繼續執行這裡的程式碼
	std::cout << "Parallel region finished." << std::endl;
	//Example usage of the Tensor class
    Tensor t1({ 2, 3, 4 });
    t1.print(t1.size());
    t1.get({ 1, 0, 2 }) = 5.0;
    std::cout << "t1.get({1, 0, 2}): " << t1.get({ 1, 0, 2 }) << std::endl;

    Tensor t2({ 5 }, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    t2.print();
    std::cout << "t2.get({3}): " << t2.get({ 3 }) << std::endl;


	//Example usage of the model class
	
	model m;
	m.add_layer(new convolution(3, 1000, 20, 100));//3x1000x1000 -> 20x901x901
	m.add_layer(new relu_3D(20, 901));//20x901x901->20x901x901
	m.add_layer(new pooling(20, 901, 3));//20x901x901->20x300x300
	m.add_layer(new convolution(20, 300, 10, 50));//20x300x300->10x251x251
	m.add_layer(new relu_3D(10, 251));//10x251x251->10x251x251
	m.add_layer(new pooling(10, 251, 2));////10x251x251->10x125x125
	m.add_layer(new convolution(10, 125, 5, 25));//10x125x125->5x101x101
	m.add_layer(new relu_3D(5, 101));//5x101x101->5x101x101
	m.add_layer(new pooling(5, 101, 2));//5x101x101->5x50x50
	m.add_layer(new flatten_3D(5, 50));//5x50x50->5*50*50
	m.add_layer(new dense(5 * 50 * 50, 5));//5*50*50->5
	m.add_layer(new relu_1D(5, 5));//5->5
	m.add_layer(new dense(5, 5));////5->5
	m.add_layer(new relu_1D(5, 5));//5->5
	m.add_layer(new dense(5, 5));//5->5
	cout << "---------compiling model---------" << endl;
	m.compile_model();
	cout << "---------compiling model end---------" << endl;
	// Create input tensor
	Tensor input({ 3, 1000, 1000 });
	// Initialize input tensor with random values
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 1000; ++j) {
			for (int k = 0; k < 1000; ++k) {
				input.get({ i, j, k }) = random_normal_float(0.0f, 1.0f);
			}
		}
	}

	cout << "gaga" << endl;
	double t = omp_get_wtime();
	Tensor output = m.forward(input);
	t = omp_get_wtime() - t;
	cout << "forward time: " << t << "s" << endl;
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


    return 0;
}
