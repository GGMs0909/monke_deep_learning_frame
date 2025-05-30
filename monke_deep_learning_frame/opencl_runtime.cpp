#include "opencl_runtime.h"


static const std::string kernel_file = "my_kernel.cl";
std::string get_kernel_file_content(const std::string& file_name) {
	std::ifstream file(file_name);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open kernel file: " << file_name << std::endl;
		// �o�̤����Ӫ��� exit�A�ӬO�ߥX���`�Ϊ�^�Ŧr�Ŧ�A���եΪ̳B�z
		throw std::runtime_error("�L�k���} Kernel ���: " + file_name);
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();
	std::string content = buffer.str();
	if (content.empty()) {
		std::cerr << "Error: Kernel file is empty: " << file_name << std::endl;
		throw std::runtime_error("Kernel ��󬰪�: " + file_name);
	}
	return content; // ��^ std::string�A�ͩR�g���ѽեΪ̺޲z
}

static void check_error(cl_int err, const char* operation) {
	if (err != CL_SUCCESS) {
		std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
		exit(EXIT_FAILURE);
	}
}

opencl_runtime& opencl_runtime::getInstance() {
	static opencl_runtime instance; // �����R�A�ܼƪ������]�ݭn����
	return instance;
}

opencl_runtime::opencl_runtime() {
	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);
	check_error(err, "cl::Platform::get");
	if (platforms.empty()) {
		std::cerr << "Error: No OpenCL platforms found." << std::endl;
		exit(EXIT_FAILURE);
	}
	platform = platforms[0];
	err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	check_error(err, "platform.getDevices");
	if (devices.empty()) {
		std::cerr << "Error: No OpenCL devices found." << std::endl;
		exit(EXIT_FAILURE);
	}
	context = cl::Context(devices, nullptr, nullptr, nullptr, &err);
	check_error(err, "cl::Context constructor");
	queue = cl::CommandQueue(context, devices[0], 0, &err);
	check_error(err, "cl::CommandQueue constructor");
	std::string kernel_source = get_kernel_file_content(kernel_file);
	
	cl::Program::Sources sources(1, std::make_pair(kernel_source.c_str(), kernel_source.length()));
	program = cl::Program(context, sources, &err);
	check_error(err, "cl::Program constructor");
	err = program.build(devices);
	check_error(err, "program.build");
	std::cout << "OpenCL runtime initialized successfully." << std::endl;
	//end
}
opencl_runtime::~opencl_runtime() {
	// Destructor logic if needed
}

cl::Context& opencl_runtime::get_context() {
	return context;
}
cl::Program& opencl_runtime::get_program() {
	return program;
}
cl::CommandQueue& opencl_runtime::get_queue() {
	return queue;
}
// End of opencl_runtime.cpp
