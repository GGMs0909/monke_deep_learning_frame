//kernel.cl
//universal 

__kernel void gradient_decent(__global float* V, __global float* grad_V, float learning_rate, int size) {
	int id = get_global_id(0);
	if (id < size) {
		float g = grad_V[id];
		if(g > 10) g = 10;
		if(g < -10) g = -10;
		V[id] -= learning_rate * grad_V[id];

	}
}
__kernel void reset_gradient( __global float* grad_V,  int size){
	int id = get_global_id(0);
	if(id < size){
		grad_V[id] = 0;
	}
}
__kernel void adam_update(
    __global float* param,
    __global float* grad,
    __global float* m,
    __global float* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int t, // Current timestep
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        float g = grad[gid]; // Current gradient
	if(g > 10) g = 10;
	if(g < -10) g = -10;

        // Update biased first moment estimate
        m[gid] = beta1 * m[gid] + (1.0f - beta1) * g;

        // Update biased second raw moment estimate
        v[gid] = beta2 * v[gid] + (1.0f - beta2) * g * g; // g * g for element-wise square

        // Compute bias-corrected first moment estimate
        float m_hat = m[gid] / (1.0f - pow(beta1, (float)t)); // pow(base, exponent)

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[gid] / (1.0f - pow(beta2, (float)t));

        // Update parameters
        param[gid] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}
__kernel void relu_forward(__global float* input, __global float* output, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        output[id] = (x > 0.0f) ? x : 0.01f * x;
    }
}
__kernel void relu_backward(__global float* grad_output, __global float* input, __global float* grad_input, int size) {
    int id = get_global_id(0);
    if (id < size) {
        grad_input[id] = (input[id] > 0.0f) ? grad_output[id] : 0.01f * grad_output[id];
    }
}
__kernel void softmax_forward(__global float* input, __global float* output, int size) {
	int id = get_global_id(0);
	if (id < size) {
		float max_val = input[0];
		for (int i = 1; i < size; i++) {
			max_val = max(max_val, input[i]);
		}
		
		float sum_exp = 0.0f;
		for (int i = 0; i < size; i++) {
			sum_exp += exp(input[i] - max_val);
		}
		
		output[id] = exp(input[id] - max_val) / sum_exp;
	}
}
__kernel void softmax_backward(__global float* grad_output, __global float* output, __global float* grad_input, int size) {
	int id = get_global_id(0);
	if (id < size) {
		float sum = 0.0f;
		for (int i = 0; i < size; i++) {
			sum += grad_output[i] * output[i];
		}
		grad_input[id] = output[id] * (grad_output[id] - sum);
	}
}
__kernel void normalization_backward(__global float* grad_parameters, __global float* parameters, int size, float normalization_factor) {
	int id = get_global_id(0);
	if (id < size) {
		grad_parameters[id] = parameters[id]* normalization_factor;
	}
}

//specific
__kernel void dense_forward(__global float* input, __global float* output, __global float* weights, __global float* biases, int input_size, int output_size) {
	int id = get_global_id(0);
	if (id < output_size) {
		float sum = 0.0f;
		for (int i = 0; i < input_size; i++) {
			sum += input[i] * weights[id * input_size + i];
		}
		output[id] = sum + biases[id];
	}
}
__kernel void dense_backward_wb(__global float* grad_output, __global float* input, __global float* grad_weights, __global float* grad_biases, int input_size, int output_size) {
	int id = get_global_id(0);
	if (id < output_size) {
		for (int i = 0; i < input_size; i++) {
			grad_weights[id * input_size + i] += grad_output[id] * input[i];
		}
		grad_biases[id] += grad_output[id];
	}
}
__kernel void dense_backward_input(__global float* grad_output, __global float* weights, __global float* grad_input, int input_size, int output_size) {
	int id = get_global_id(0);
	if (id < input_size) {
		float sum = 0.0f;
		for (int i = 0; i < output_size; i++) {
			sum += grad_output[i] * weights[i * input_size + id];
		}
		grad_input[id] = sum;
	}
}
__kernel void convolution_forward(__global float* input, __global float* output, __global float* weights, __global float* biases, int input_channels, int input_size, int output_channels, int kernel_size){
	int h = get_global_id(0);
	int r = get_global_id(1);
	int c = get_global_id(2);
	int output_size = (input_size - kernel_size + 1);
	
	if (h < output_channels && r < output_size && c < output_size) {
		float sum = 0.0f;
		for (int i = 0; i < input_channels; i++) {
			for (int j = 0; j < kernel_size; j++) {
				for(int k = 0; k < kernel_size; k++){
					int weights_index = h * (input_channels * kernel_size * kernel_size) + i * (kernel_size * kernel_size) + j * kernel_size + k;
					//weights[h][i][j][k]
					int input_index = i * (input_size * input_size) + (r + j) * input_size + (c + k);
					//input[i][r + j][c + k]
					sum += input[input_index] * weights[weights_index];
				}
			}
		}
		output[h * (output_size * output_size) + r * output_size + c] = sum + biases[h];
	}
}

__kernel void convolution_backward_weights(__global float* grad_output, __global float* input, __global float* grad_weights, int input_channels, int input_size, int output_channels, int kernel_size){
	int h = get_global_id(0);
	int i = get_global_id(1);
	int jk = get_global_id(2);
	if(h < output_channels && i < input_channels && jk < kernel_size*kernel_size) {
		//jk = j * kernel_size + k
		int j = jk / kernel_size; // row in kernel
		int k = jk % kernel_size; // column in kernel
		float sum = 0.0f;
		int output_size = (input_size - kernel_size + 1);
		for(int r = 0; r < output_size; r++) {
			for(int c = 0; c < output_size; c++) {
				int input_index = i * (input_size * input_size) + (r + j) * input_size + (c + k);
				int grad_output_index = h * (output_size * output_size) + r * output_size + c;
				sum += input[input_index] * grad_output[grad_output_index];
			}
		}
		grad_weights[h * (input_channels * kernel_size * kernel_size) + i * (kernel_size * kernel_size) + j * kernel_size + k] += sum;
	}
}
__kernel void convolution_backward_biases(__global float* grad_output, __global float* grad_biases, int output_channels, int output_size){
	int h = get_global_id(0);
	if (h < output_channels) {
		float sum = 0.0f;
		for (int r = 0; r < output_size; r++) {
			for (int c = 0; c < output_size; c++) {
				int grad_output_index = h * (output_size * output_size) + r * output_size + c;
				sum += grad_output[grad_output_index];
			}
		}
		grad_biases[h] += sum;
	}
}
__kernel void convolution_backward_input(__global float* grad_output, __global float* weights, __global float* grad_input, int input_channels, int input_size, int output_channels, int kernel_size) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	
	if(i < input_channels && j < input_size && k < input_size){
		float sum = 0;
        int output_size = (input_size - kernel_size + 1);


		for (int h = 0; h < output_channels; h++) {
			for (int r = 0; r < kernel_size; r++) {
				for (int c = 0; c < kernel_size; c++) {
                    int r_o = j - r;
                    int c_o = k - c;

					if (r_o >= 0 && r_o < output_size && c_o >= 0 && c_o < output_size) {
						int grad_output_index = h * (output_size * output_size) + r_o * output_size + c_o;
						int weights_index = h * (input_channels * kernel_size * kernel_size) + i * (kernel_size * kernel_size) + r * kernel_size + c;
						sum += grad_output[grad_output_index] * weights[weights_index];
					}
				}
			}
		}

		int grad_input_index = i * (input_size * input_size) + j * input_size + k;
		grad_input[grad_input_index] = sum;
	}
}

__kernel void pooling_forward(__global float* input, __global float* output, __global int* max_indices, int input_channels, int input_size, int pool_size){
	int h = get_global_id(0);
	int r = get_global_id(1);
	int c = get_global_id(2);
	int output_channels = input_channels;
	int output_size = input_size / pool_size;
	if(h < input_channels && r < output_size && c < output_size){
		float max_val = input[h*(input_size*input_size) + (r*pool_size)*input_size + (c*pool_size)];
		//input[h][r*pool_size][c*pool_size]
		int max_i = 0;
		int max_j = 0;
		for(int i = 0; i < pool_size; ++i){
			for(int j = 0; j < pool_size; ++j){
				int input_index = h*(input_size*input_size) + (r*pool_size+i)*input_size + (c*pool_size+j);
				//input[h][r*pool_size+i][c*pooling_size+j]
				if(input[input_index] > max_val){
					max_val = input[input_index];
					max_i = i;
					max_j = j;
				}
			}
		}
		int output_index = h*(output_size*output_size) + r*(output_size) + c;
		//output[h][r][c]
		output[output_index] = max_val;
		int max_indices_index = h*(output_size*output_size*2) + r*(output_size*2) + c*2;
		//max_indices[h][r][c][0/1]
		max_indices[max_indices_index] = max_i;
		max_indices[max_indices_index+1] = max_j;
	}
}
__kernel void pooling_backward(__global float* grad_output, __global float* grad_input, __global int* max_indices, int input_channels, int input_size, int pool_size){
	int h = get_global_id(0);
	int r = get_global_id(1);
	int c = get_global_id(2);
	int output_channels = input_channels;
	int output_size = input_size / pool_size;
	if(h < input_channels && r < output_size && c < output_size){
		int max_indices_index = h*(output_size*output_size*2) + r*(output_size*2) + c*2;
		int max_i = max_indices[max_indices_index];
		int max_j = max_indices[max_indices_index+1];

		int input_index = h*(input_size*input_size) + (r*pool_size+max_i)*input_size + (c*pool_size+max_j);
		int output_index = h*(output_size*output_size) + r*(output_size) + c;
		grad_input[input_index] += grad_output[output_index];
	}
}

__kernel void scale_forward(__global float* input, __global float* output, int size, int scale) {
	int id = get_global_id(0);
	if (id < size) {
		output[id] = input[id] / scale;
	}
}
__kernel void scale_backward(__global float* grad_output, __global float* grad_input, int size, int scale) {
	int id = get_global_id(0);
	if (id < size) {
		grad_input[id] = grad_output[id] * scale;
	}
}


