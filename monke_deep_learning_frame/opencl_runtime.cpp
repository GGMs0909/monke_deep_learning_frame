#include "opencl_runtime.h"
#include <stdexcept> // for std::runtime_error


static const std::string kernel_file_name = "kernel.cl";


std::string get_kernel_file_content(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << file_name << std::endl;
        throw std::runtime_error("cannot open kernel file: " + file_name);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    std::string content = buffer.str();
    if (content.empty()) {
        std::cerr << "Error: Kernel file is empty: " << file_name << std::endl;
        throw std::runtime_error("Kernel empty: " + file_name);
    }
    return content;
}


opencl_runtime& opencl_runtime::getInstance() {
    static opencl_runtime instance;
    return instance;
}


void opencl_runtime::check_error(cl_int err_code, const char* operation) {
    if (err_code != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err_code << std::endl;

        throw std::runtime_error(std::string("OpenCL Error: ") + operation + " failed with code " + std::to_string(err_code));
    }
}


opencl_runtime::opencl_runtime() : is_initialized_flag(false), err(CL_SUCCESS) {

}

opencl_runtime::~opencl_runtime() {

}


void opencl_runtime::initialize() {
    if (is_initialized_flag) {
        std::cout << "opencl_runtime was already initialized" << std::endl;
        return;
    }

    try {

        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        check_error(err, "cl::Platform::get");
        if (platforms.empty()) {
            throw std::runtime_error("cannot find any opencl platform");
        }
        platform = platforms[0]; 

  
        std::vector<cl::Device> devices_found;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_found);
        check_error(err, "platform.getDevices (GPU)");
        if (devices_found.empty()) {
            std::cerr << "cannot find OpenCL device" << std::endl;
            err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_found);
            check_error(err, "platform.getDevices (ALL)");
        }
        if (devices_found.empty()) {
            throw std::runtime_error("cannot find any opencl device");
        }
        device = devices_found[0];
        std::cout << "found OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "device max compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;

       
        context = cl::Context(device, nullptr, nullptr, nullptr, &err);
        check_error(err, "cl::Context constructor");


        queue = cl::CommandQueue(context, device, 0, &err); 
        check_error(err, "cl::CommandQueue constructor");

 
        std::string kernel_source = get_kernel_file_content(kernel_file_name);
        cl::Program::Sources sources(1, std::make_pair(kernel_source.c_str(), kernel_source.length()));
        program = cl::Program(context, sources, &err);
        check_error(err, "cl::Program constructor");


        err = program.build(devices_found); 
        if (err != CL_SUCCESS) {
            std::string build_log;

            program.getBuildInfo(devices_found[0], CL_PROGRAM_BUILD_LOG, &build_log);
            std::cerr << "OPENCL KERNEL compile failed!\n";
            std::cerr << "device (" << devices_found[0].getInfo<CL_DEVICE_NAME>() << ") 's log:\n" << build_log << std::endl;
            check_error(err, "program.build"); 
        }

        is_initialized_flag = true;
        std::cout << "OpenCL intialize succesed!!!" << std::endl;

    }
    catch (const std::runtime_error& e) {
        std::cerr << "OpenCL intialize fail!!!" << e.what() << std::endl;
        is_initialized_flag = false; 
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "OpenCL intialize fail!!!" << e.what() << std::endl;
        is_initialized_flag = false;
        throw std::runtime_error("opencl_runtime initia;ize fail : unknowned error");
    }
}


cl::Context& opencl_runtime::get_context() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime is not initialize()");
    }
    return context;
}

cl::Program& opencl_runtime::get_program() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime is not initialize()");
    }
    return program;
}

cl::CommandQueue& opencl_runtime::get_queue() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime is not initialize()");
    }
    return queue;
}

cl::Device& opencl_runtime::get_device() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime is not initialize()");
    }
    return device;
}