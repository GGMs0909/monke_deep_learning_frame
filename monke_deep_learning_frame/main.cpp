#include "model.h"
#include <vector>
#include <omp.h>  // For omp_get_wtime()
#include <random>
using namespace std;

//Relu(int input_size);
//Softmax(int input_size);
//Dense(int input_size, int output_size);
//Convolution(int input_channels, int input_size, int output_channels, int kernel_size);
//Pooling(int input_channels, int input_size, int pool_size);
//Flatten_3D(int input_channels, int input_size);
static float random_normal_float(float mean, float stddev) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<float> dist(mean, stddev);
	return dist(gen);
}


int main() {
	string LOTOFLINE = "----------------------";
	// Initialize OpenCL runtime

	opencl_runtime::getInstance().initialize();

	cout << "-----------" << endl;

	Model M;
	// Add layers to the model
	M.add_layer(new Convolution(1, 28, 3, 3));//1x28x28 -> 3x26x26
	M.add_layer(new Relu(26 * 26 * 3)); // Activation layer after convolution
	M.add_layer(new Pooling(3, 26, 2)); // Pooling layer with 2x2 pooling size
	M.add_layer(new Flatten_3D(3, 13)); // Flatten the pooled output (3x13x13) to a 1D vector
		
	M.add_layer(new Dense(3*13*13, 128)); // Dense layer with 128 neurons
	M.add_layer(new Relu(128)); // Activation layer
	M.add_layer(new Dense(128, 10)); // Output layer with 10 classes (e.g., MNIST digits)
	M.add_layer(new Softmax(10)); // Softmax activation for output layer

	// Compile the model with input shape, loss function, and optimizer
	M.compile({ 1,28,28 }, *new LossWithNormalization(&M, new CrossEntropyLoss(10), 1e-3), *new Adam(0.9, 0.99, 1e-08));

	cout << "Model compiled successfully." << endl;
	cout << LOTOFLINE << endl;
	Tensor input({ 1,28,28 }); // Example input tensor
	Tensor output({ 10 }); // Example output tensor
	Tensor real({ 10 },{1,0,0,0,0,0,0,0,0,0}); // Example real tensor (ground truth)

	for (float& x : input.data) {
		x = random_normal_float(0.0f, 1.0f); // Fill input tensor with random values
	}
	for (int i = 0; i < 10; i++) {
		float loss = M.forward_with_loss(input, output, real); // Forward pass with loss calculation
		cout << "Loss: " << loss << endl; // Print the calculated loss
		M.backward(output, real); // Backward pass through the model
		cout << "Gradient Norm: " << M.get_gradient_norm() << endl; // Print the gradient norm
		M.update(); // Update model parameters using the optimizer
	}

	M.print_parameters(); // Print model parameters for debugging

	cout << "Model updated successfully." << endl;
	cout << "predict after training" << endl;
	output.print(10); // Print the first 10 elements of the output tensor
	cout << "real" << endl;
	real.print(10); // Print the first 10 elements of the real tensor (ground truth)
	cout << LOTOFLINE << endl;


}