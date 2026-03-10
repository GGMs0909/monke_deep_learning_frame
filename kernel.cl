


__kernel void set_to_zero(__global float* A, int size){
    int i = get_global_id(0);

    if(i < size){
        A[i] = 0;
    }
}

__kernel void reduction(__global const float* input, __global float* output, __local float* scratch, int size){
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int gsize = get_local_size(0);
    int id = get_global_id(0);

    scratch[lid] = (id < size) ? input[id] : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);


    for(int stride = gsize / 2; stride > 0; stride >>= 1){
        if(lid < stride){
            scratch[lid] += scratch[lid + stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        output[gid] = scratch[0];
    }

}



//Linear algebra

__kernel void vector_combine(__global const float* A, __global const float* B, __global float* C, int N, float a, float b){
    int i = get_global_id(0);

    if(i < N){
        C[i] = a*A[i] + b*B[i];
    }
}



//loss
//MSE
__kernel void MSE_forward(__global const float* pred, __global const float* target, __global float* output, int size){
    int id = get_global_id(0);

    if(id < size){
        output[id] = (pred[id]-target[id])*(pred[id]-target[id]);
    }
}
__kernel void MSE_backward(__global const float* pred, __global const float* target, __global float* grad_inputs, int size){
    int id = get_global_id(0);

    if(id < size){
        grad_inputs[id] = 2*(pred[id] - target[id]) / size;
    }
}
//crossentropy
__kernel void crossentropy_forward(__global const float* pred, __global const float* target, __global float* output, int size){
    int id = get_global_id(0);

    if(id < size){
        output[id] = -target[id] * log(pred[id] + 1e-8f);
    }
}

__kernel void crossentropy_backward(__global const float* pred, __global const float* target, __global float* grad_inputs, int size, int batch_size){
    int id = get_global_id(0);

    if(id < size*batch_size){
        grad_inputs[id] = -target[id] / (pred[id] + 1e-8f) / batch_size;
    }
}

//optimizers
//gd
__kernel void gd_update(__global float* params, __global float* grad_params, float learning_rate, int size){
    int i = get_global_id(0);
    
    if(i < size){
        float g = grad_params[i];
        if(g > 1) g = 1;
        if(g < -1) g = -1;

        params[i] -= learning_rate*g;
    }
}
//adam
__kernel void adam_update(__global float* params, __global const float* grad_params, __global float* m, __global float* v, float beta1, float beta2, float epsilon, float beta1t, float beta2t, float learning_rate, int size){
    int i = get_global_id(0);

    if(i < size){
        float g = grad_params[i];
        if(g > 1) g = 1;
        if(g < -1) g = -1;
        m[i] = beta1*m[i] + (1-beta1)*g;
        v[i] = beta2*v[i] + (1-beta2)*g*g;
        float m_hat = m[i]/(1-beta1t);
        float v_hat = v[i]/(1-beta2t);

        params[i] -= learning_rate*m_hat / (sqrt(v_hat) + epsilon);
    }
}



//calculate function
//scale
__kernel void scale_forward(__global const float* inputs, __global float* outputs, int size, float scale_factor){
    int i = get_global_id(0);
    
    if(i < size){
        outputs[i] = inputs[i]*scale_factor;
    }
}
__kernel void scale_backward(__global const float* grad_outputs, __global float* grad_inputs, int size, float scale_factor){
    int i = get_global_id(0);

    if(i < size){
        grad_inputs[i] = grad_outputs[i]*scale_factor;
    }
}
//dropout
uint hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__kernel void dropout_forward(__global const float* inputs,
                              __global float* outputs,
                              __global float* masks,
                              float p,
                              uint seed,
                              int size)
{
    int i = get_global_id(0);

    if(i < size){
        uint thread_seed = hash(seed + i);
        float r = (float)thread_seed / (float)0xFFFFFFFF;

        if(r < p){
            outputs[i] = 0.0f;
            masks[i] = 0.0f;
        }
        else{
            outputs[i] = inputs[i] / (1-p);
            masks[i] = 1.0f;
        }
    }
}
__kernel void dropout_backward(__global const float* grad_outputs,
                               __global const float* masks,
                               __global float* grad_inputs,
                               float p,
                               int size)
{
    int i = get_global_id(0);

    if(i < size){
        if(masks[i] < 0.5){
            grad_inputs[i] = 0.0f;
        }
        else{
            grad_inputs[i] = grad_outputs[i] / (1-p);
        }
    }
}


//dense
__kernel void dense_forward(__global const float* inputs,
                            __global float* outputs,
                            __global const float* weights,
                            __global const float* biases,
                            int batch_size, int input_size, int output_size)
{
    int i = get_global_id(0);
    int b = get_global_id(1);
    
    if(i < output_size && b < batch_size){
        float sum = 0;
        for(int j = 0; j < input_size; j++){
            sum += weights[i*input_size + j] * inputs[b*input_size + j];
        }
        outputs[b*output_size + i] = sum + biases[i];
    }
}
__kernel void dense_forward2(__global const float* inputs,
                            __global float* outputs,
                            __global const float* weights,
                            __global const float* biases,
                            int batch_size, int input_size, int output_size)
{
    int ylid = get_local_id(0);//0 ~ 15
    int blid = get_local_id(1);//0 ~ 15
    int ygid = get_group_id(0);
    int bgid = get_group_id(1);
    int i = get_global_id(0);
    int b = get_global_id(1);
    __local float cacheW[16][16];
    __local float cacheI[16][16];

    float sum = 0.0f;
    
    for(int offset = 0; offset < input_size; offset += 16){
        //cacheW[ylid][blid] = weights[i][offset + blid];
        //cacheI[blid][ylid] = inputs[b][offset + ylid];
        if(i < output_size && offset + blid < input_size){
            cacheW[ylid][blid] = weights[i*input_size + (offset + blid)];
        }
        else cacheW[ylid][blid] = 0.0f;
        if(b < batch_size && offset + ylid < input_size){
            cacheI[blid][ylid] = inputs[b*input_size + (offset + ylid)];
        }
        else cacheI[blid][ylid] = 0.0f;
        
        
        barrier(CLK_LOCAL_MEM_FENCE);

        //output[b][i] += cacheW[ylid][j]*cacheI[blid][j];
        
        for(int k = 0; k < 16; k++){
            sum += cacheW[ylid][k]*cacheI[blid][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(i < output_size && b < batch_size){
        outputs[b*output_size + i] = sum + biases[i];
    }
        
    
}

__kernel void dense_backward_wb(__global const float* grad_outputs,
                                __global const float* inputs,
                                __global float* grad_weights,
                                __global float* grad_biases,
                                int batch_size, int output_size, int input_size)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if(i < output_size && j < input_size){
        if(j == 0){
            float sum = 0;
            for(int b = 0; b < batch_size; b++){
                sum += grad_outputs[b*output_size + i];
            }
            grad_biases[i] = sum;
        }
        float sum = 0;
        for(int b = 0; b < batch_size; b++){
            sum += grad_outputs[b*output_size + i]*inputs[b*input_size + j];
        }
        grad_weights[i*input_size + j] = sum;
    }
}

__kernel void dense_backward_inputs(__global const float* grad_outputs,
                                    __global const float* weights,
                                    __global float* grad_inputs,
                                    int batch_size, int output_size, int input_size)
{   
    int b = get_global_id(1);
    int i = get_global_id(0);
    

    if(b < batch_size && i < input_size){
        float sum = 0;
        for(int j = 0; j < output_size; j++){
            sum += grad_outputs[b*output_size + j] * weights[j*input_size + i];
        }
        grad_inputs[b*input_size + i] = sum;
    }
}
//convolution
__kernel void convolution_forward(__global const float* inputs,
                                  __global float* outputs,
                                  __global const float* weights,
                                  __global const float* biases,
                                  int batch_size,
                                  int input_channels,
                                  int input_size,
                                  int output_channels,
                                  int output_size,
                                  int kernel_size)
{
    int jk = get_global_id(0);
    int i = get_global_id(1);
    int b = get_global_id(2);
    if(jk < output_size*output_size && i < output_channels && b < batch_size){
        //outputs[b][i][j][k] = b*output_channels*output_size*output_size + i*output_size*output_size + j*output_size + k;
        //weights[i][h][r][c] = i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size + r*kernel_size + c;
        //inputs[b][h][j+r][k+c] = b*input_channels*input_size*input_size + h*input_size*input_size + (j+r)*input_size + (k+c);
        int k = jk%output_size;
        int j = (jk/output_size);
        int outputs_index = b*output_channels*output_size*output_size + i*output_size*output_size + j*output_size + k;
        float sum = 0;
        for(int h = 0; h < input_channels; h++){
            for(int r = 0; r < kernel_size; r++){
                for(int c = 0; c < kernel_size; c++){
                    int weights_index = i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size + r*kernel_size + c;
                    int inputs_index = b*input_channels*input_size*input_size + h*input_size*input_size + (j+r)*input_size + (k+c);
                    sum += weights[weights_index]*inputs[inputs_index];
                }
            }
        }
        outputs[outputs_index] = sum + biases[i];
    }
}

// __kernel void convolution_forward2(__global const float* inputs,
//                                   __global float* outputs,
//                                   __global const float* weights,
//                                   __global const float* biases,
//                                   int batch_size,
//                                   int input_channels,
//                                   int input_size,
//                                   int output_channels,
//                                   int output_size,
//                                   int kernel_size)
// {
//     int klid = get_local_id(0);
//     int jlid = get_local_id(1);
//     int kgid = get_group_id(0);
//     int jgid = get_group_id(1);
//     int k = get_global_id(0);
//     int j = get_global_id(1);
//     int bi = get_global_id(2); // b*output_channels + i
//     __local float cacheW[16][16];
//     __local float cacheI[16][16];

//     float sum = 0.0f;
//     //cache[]
//     for(int h = 0; h < input_channels; h++){
//         for(int Ioffset1 = jgid*16; Ioffset1 < jgid*16 + 16 + kernel_size - 1; Ioffset1 += 16){
//             for(int Ioffset2 = kgid*16; Ioffset2 < kgid*16 + 16 + kernel_size - 1; Ioffset2 += 16){

//                 for(int )
//             }
//         }
//     }

    
// }
__kernel void convolution_backward_weights(__global const float* grad_outputs,
                                           __global const float* inputs,
                                           __global float* grad_weights,
                                           int batch_size,
                                           int input_channels,
                                           int input_size,
                                           int output_channels,
                                           int output_size,
                                           int kernel_size)
{
    int hrc = get_global_id(0);
    int i = get_global_id(1);
    

    if(hrc < input_channels*kernel_size*kernel_size && i < output_channels){
        //weights[i][h][r][c] = i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size + r*kernel_size + c;
        //grad_outputs[b][i][j][k] = b*output_channels*output_size*output_size + i*output_size*output_size + j*output_size + k;
        //inputs[b][h][j+r][k+c] = b*input_channels*input_size*input_size + h*input_size*input_size + (j+r)*input_size + k+c;

        int c = hrc%kernel_size;
        int r = (hrc/kernel_size)%kernel_size;
        int h = hrc/(kernel_size*kernel_size);

        
        float sum = 0;
        for(int b = 0; b < batch_size; b++){
            int grad_outputs_index_offset = b*output_channels*output_size*output_size + i*output_size*output_size;
            int inputs_index_offset = b*input_channels*input_size*input_size + h*input_size*input_size;
            for(int j = 0; j < output_size; j++){
                for(int k = 0; k < output_size; k++){
                    sum += grad_outputs[grad_outputs_index_offset + j*output_size + k] * inputs[inputs_index_offset + (j+r)*input_size + k+c];
                }
            }
        }
        grad_weights[i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size + r*kernel_size + c] = sum;

    }
}
__kernel void convolution_backward_biases(__global const float* grad_outputs,
                                          __global float* grad_biases,
                                          int batch_size,
                                          int output_channels,
                                          int output_size)
{
    int i = get_global_id(0);
    

    if(i < output_channels){
        //grad_outputs[b][i][j][k] = b*output_channels*output_size*output_size + i*output_size*output_size + j*output_size + k;
        


        
        
        float sum = 0;
        for(int b = 0; b < batch_size; b++){

            int grad_outputs_index_offset = b*output_channels*output_size*output_size + i*output_size*output_size;

            for(int j = 0; j < output_size; j++){
                for(int k = 0; k < output_size; k++){
                    sum += grad_outputs[grad_outputs_index_offset + j*output_size + k];
                }
            }
        }
        
        grad_biases[i] = sum;

    }
}
__kernel void convolution_backward_inputs(__global const float* grad_outputs,
                                          __global const float* weights,
                                          __global float* grad_inputs,
                                          int batch_size,
                                          int input_channels,
                                          int input_size,
                                          int output_channels,
                                          int output_size,
                                          int kernel_size)
{
    int xy = get_global_id(0);
    int h = get_global_id(1);
    int b = get_global_id(2);
    if(xy < input_size*input_size && h < input_channels && b < batch_size){
        int x = xy/input_size;
        int y = xy%input_size;
        float sum = 0;
        //grad_outputs[b][i][x-r][y-c] = b*output_channels*output_size*output_size + i*output_size*output_size + (x-r)*output_size + y-c;
        //weights[i][h][r][c] = i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size + r*kernel_size + c;
        for(int i = 0; i < output_channels; i++){
            int grad_outputs_index_offset =  b*output_channels*output_size*output_size + i*output_size*output_size;
            int weights_index_offset = i*input_channels*kernel_size*kernel_size + h*kernel_size*kernel_size;
            for(int r = 0; r < kernel_size; r++){
                for(int c = 0; c < kernel_size; c++){
                    if(x-r >= 0 && x-r < output_size && y-c >= 0 && y-c < output_size){
                        sum += grad_outputs[grad_outputs_index_offset + (x-r)*output_size + y-c]*weights[weights_index_offset + r*kernel_size + c];
                    }
                }
            }
        }
        grad_inputs[b*input_channels*input_size*input_size + h*input_size*input_size + xy] = sum;
    }
}


//activation function
//relu
__kernel void relu_forward(__global const float* inputs,
                           __global float* outputs,
                           int size, float slope)
{
    int i = get_global_id(0);

    if(i < size){
        outputs[i] = (inputs[i] > 0 ) ? inputs[i] : (inputs[i]*slope);   
    }
}
__kernel void relu_backward(__global const float* grad_outputs,
                            __global const float* inputs,
                            __global float* grad_inputs,
                            int size, float slope)
{
    int i = get_global_id(0);

    if(i < size){
        grad_inputs[i] = grad_outputs[i] * ((inputs[i] > 0) ? 1.0f : slope);
    }
}
//softmax
__kernel void softmax_forward(__global const float* inputs,
                              __global float* outputs,
                              int batch_size, int size)
{
    int i = get_global_id(0);
    int b = get_global_id(1);

    if(i < size && b < batch_size){
        float max_val = -1e38f; 
        for(int j = 0; j < size; j++){
            if(inputs[b*size + j] > max_val) max_val = inputs[b*size + j];
        }

        // 2. 計算 sum (減去 max_val)
        float sum = 0;
        for(int j = 0; j < size; j++){
            sum += exp(inputs[b*size + j] - max_val);
        }

        outputs[b*size + i] = exp(inputs[b*size + i] - max_val) / sum;
    }
}
__kernel void softmax_backward(__global const float* grad_outputs,
                               __global const float* outputs,
                               __global float* grad_inputs,
                               int batch_size, int size)
{
    int i = get_global_id(0);
    int b = get_global_id(1);

    if(i < size && b < batch_size){
        float sum = 0;
        for(int j = 0; j < size; j++){
            sum += grad_outputs[b*size + j]*outputs[b*size+j];
        }
        grad_inputs[b*size + i] = outputs[b*size+i]*(grad_outputs[b*size+i] - sum);
    }
}

//normalization function

//batch_normalization

__kernel void BN_forward_mean_sqrtVar(__global const float* inputs,
                              __global float* means,
                              __global float* sqrtVars,
                              __global float* AvMeans,
                              __global float* AvSVs,
                              float p,
                              int size,
                              int batch_size)
{
    int i = get_global_id(0);
    if(i < size){
        float sum = 0.0f;
        float sumSq = 0.0f;

        for(int b = 0; b < batch_size; b++){
            float x = inputs[b*size + i];
            sum += x;
            sumSq += x*x;
        }
        float m = sum / batch_size;
        float var = (sumSq / batch_size) - (m * m);
        means[i] = m;
        sqrtVars[i] = sqrt(var + 1e-5f);

        AvMeans[i] = p * m + (1.0f - p) * AvMeans[i];
        AvSVs[i] = p * sqrtVars[i] + (1.0f - p) * AvSVs[i];

        
    }
}

__kernel void BN_forward(__global const float* inputs,
                         __global float* outputs,
                         __global float* means,
                         __global float* sqrtVars,
                         __global float* gammas,
                         __global float* betas,
                         int size,
                         int batch_size)
{
    int i = get_global_id(0);
    int b = get_global_id(1);

    if(i < size && b < batch_size){
        outputs[b*size + i] = (inputs[b*size + i]-means[i])/sqrtVars[i] *gammas[i] + betas[i];
    }
}

__kernel void BN_backward_gb(__global const float* grad_outputs,
                             __global const float* inputs,
                             __global const float* means,
                             __global const float* sqrtVars,
                             __global float* grad_gammas,
                             __global float* grad_betas,
                             int size,
                             int batch_size
                             )
{
    int i = get_global_id(0);

    if(i < size){
        float sumg = 0.0f;
        float sumb = 0.0f;
        float mean = means[i];
        float sqrtVar = sqrtVars[i];
        for(int b = 0; b < batch_size; b++){
            float g_o = grad_outputs[b*size + i];
            sumg += g_o*(inputs[b*size + i] - mean)/sqrtVar;
            sumb += g_o;
        }

        grad_gammas[i] = sumg;
        grad_betas[i] = sumb;
    }
}

__kernel void BN_backward_inputs(__global const float* grad_outputs,
                                 __global const float* inputs,
                                 __global const float* gammas,
                                 __global const float* means,
                                 __global const float* sqrtVars,
                                 __global float* grad_inputs,
                                 int size,
                                 int batch_size)
{
    int i = get_global_id(0);
    

    if(i < size){
        float sumA = 0.0f;
        float sumB = 0.0f;
        float mean = means[i];
        float sqrtVar = sqrtVars[i];
        for(int k = 0; k < batch_size; k++){
            float g_o = grad_outputs[k*size + i];
            sumA += g_o;
            sumB += (inputs[k*size + i] - mean )/ sqrtVar *g_o;
        }
        float x = gammas[i] / (batch_size * sqrtVar);
        for(int b = 0; b < batch_size; b++){
            grad_inputs[b*size + i] = x*(batch_size*grad_outputs[b*size + i] - sumA - (inputs[b*size + i]-mean)/sqrtVar * sumB);
        }
    }
}
