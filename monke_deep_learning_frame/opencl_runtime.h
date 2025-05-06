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
	std::vector<cl::Device> devices;
	cl::Program program;
	cl_int err;
	opencl_runtime();
	~opencl_runtime();
public:
	static opencl_runtime& getInstance();
	cl::Context& get_context();
	cl::Program& get_program();
	cl::CommandQueue& get_queue();
};
