#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

class opencl_runtime {
private:
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device; 
    cl::Program program;
    cl_int err; 

    bool is_initialized_flag; 


    opencl_runtime();


    ~opencl_runtime();


    opencl_runtime(const opencl_runtime&) = delete;
    opencl_runtime& operator=(const opencl_runtime&) = delete;


    void check_error(cl_int err, const char* operation);

public:

    static opencl_runtime& getInstance();


    void initialize();

    cl::Context& get_context();

    cl::Program& get_program();

    cl::CommandQueue& get_queue();

    cl::Device& get_device();
};