#include "model.h"
#include <vector>
using namespace std;

//Relu(int input_size);
//Softmax(int input_size);
//Dense(int input_size, int output_size);
//Convolution(int input_channels, int input_size, int output_channels, int kernel_size);
//Pooling(int input_channels, int input_size, int pool_size);
//Flatten_3D(int input_channels, int input_size);

int main() {
	// Initialize OpenCL runtime
	opencl_runtime::getInstance().initialize();
	cout << "-----------" << endl;
	Model M(
		*new CrossEntropyLoss(10),
		*new GradientDescent(0.01f)
	);
	M.add_layer(new Convolution(3, 1000, 10, 100));//3x1000 -> 10x901
	M.add_layer(new Relu(10 * 901));
	M.add_layer(new Pooling(10, 901, 2));//10x901 -> 10x450
	M.add_layer(new Convolution(10, 450, 20, 50));//10x450 -> 20x401
	M.add_layer(new Relu(20 * 401));
	M.add_layer(new Pooling(20, 401, 2));//20x401->20x200
	M.add_layer(new Convolution(20, 200, 10, 10));//20x200 -> 10x191
	M.add_layer(new Pooling(10, 191, 2));//10x191->10x95
	M.add_layer(new Flatten_3D(10, 95));//10x95 -> 950
	M.add_layer(new Dense(950, 500));//950->500
	M.add_layer(new Relu(500));
	M.add_layer(new Dense(500, 250));//500 -> 250
	M.add_layer(new Relu(250));
	M.add_layer(new Dense(250, 100));//250 -> 100
	M.add_layer(new Relu(100));
	M.add_layer(new Dense(100, 20));//100->20
	M.add_layer(new Relu(20));
	M.add_layer(new Dense(20, 10));
	M.add_layer(new Softmax(10));
	cout << "Model layers added." << endl;
	M.compile();
	Tensor input({ 3,1000,1000 });
	Tensor output({10});
	Tensor real({10});
	float loss_value = 0.0f;
	cout << "press enter to start forward" << endl;
	cin.get();
	M.forward(input, output, real, loss_value);
	cout << "forward done." << endl;
	cout << "Output: ";
	output.print();
	cout << "Loss: " << loss_value << endl;
	
	
	cout << "press enter to start backward" << endl;
	cin.get();
	M.backward(output, real);
	cout << "backward done." << endl;
	return 0;

}