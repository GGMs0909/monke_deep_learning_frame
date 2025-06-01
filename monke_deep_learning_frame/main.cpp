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

	Model M(

		*new CrossEntropyLoss(10),

		*new Adam(0.1f, 0.9f, 0.999f, 1e-8f)

	);

	M.add_layer(new Convolution(1, 10, 2, 5)); // 1x10x10 -> 2x6x6
	M.add_layer(new Relu(2 * 6 * 6)); // 2x6x6 -> 2x6x6
	M.add_layer(new Pooling(2, 6, 2)); // 2x6x6 -> 2x3x3
	M.add_layer(new Convolution(2, 3, 4, 3)); // 2x3x3 -> 4x1x1
	M.add_layer(new Relu(4 * 1 * 1)); // 4x1x1 -> 4x1x1
	M.add_layer(new Flatten_3D(4, 1)); // 4x1x1 -> 4
	M.add_layer(new Dense(4, 10)); // 4 -> 10
	M.add_layer(new Softmax(10)); // 10 -> 10
	

	cout << "Model layers added." << endl;

	M.compile();

	Tensor input({ 1,10,10 });

	Tensor output({ 10 });

	Tensor real({ 10 },{1,0,0,0,0,0,0,0,0,0});

	float loss_value = 0.0f;
	for (int i = 0; i < 1;++i) {
		for (int j = 0; j < 10; ++j) {
			for (int k = 0; k < 10; ++k) {
				input.get({ i, j, k }) = random_normal_float(0.0f, 1.0f);
			}
		}
	}
	input.transfer_to_gpu();
	
	cout << "performing forward pass..." << endl;
	for (int i = 0; i < 100; ++i) {
		M.forward(input, output, real, loss_value);
		cout << "Loss value: " << loss_value << endl;
		M.backward(output, real);
		M.update();
	}
	cout << "Forward pass completed." << endl;
	cout << "output: " << endl;
	output.print();



	return 0;



}