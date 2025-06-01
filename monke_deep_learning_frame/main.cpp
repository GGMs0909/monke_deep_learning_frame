#include "model.h"
#include <vector>
using namespace std;

int main() {
	// Initialize OpenCL runtime
	opencl_runtime::getInstance().initialize();
	cout << "-----------" << endl;
	Model M(
		*new MeanSquaredError(10),
		*new GradientDescent(0.01f)
	);
	M.add_layer(new Convolution())
	M.add_layer(new Dense(10, 20));
	M.add_layer(new Relu(20));
	M.add_layer(new Dense(20, 10));
	cout << "Model layers added." << endl;
	M.compile();
	Tensor input({ 10 }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });
	Tensor output({ 10 });
	Tensor real({ 10 }, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f });
	float loss_value = 0.0f;
	M.forward(input, output, real, loss_value);
	cout << "Output: ";
	output.print();
	cout << "Loss: " << loss_value << endl;

	M.forward(input, output, real, loss_value);
	cout << "Output: ";
	output.print();
	cout << "Loss: " << loss_value << endl;
	M.backward(output, real);

	return 0;

}