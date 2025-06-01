#include "opencl_runtime.h"
#include <stdexcept> // for std::runtime_error

// �R�A�`�q�AKernel ���W
static const std::string kernel_file_name = "kernel.cl";

// ���U�禡�GŪ�� Kernel ��󤺮e
std::string get_kernel_file_content(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << file_name << std::endl;
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
    return content;
}

// ��ҹ�Ҫ����
opencl_runtime& opencl_runtime::getInstance() {
    static opencl_runtime instance;
    return instance;
}

// ���U�禡�G�ˬd OpenCL ���~�éߥX���`
void opencl_runtime::check_error(cl_int err_code, const char* operation) {
    if (err_code != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err_code << std::endl;
        // �N cl_int ���~�ഫ�� std::runtime_error �ߥX�A�H�K�b main ��Ƥ��Τ@����
        throw std::runtime_error(std::string("OpenCL Error: ") + operation + " failed with code " + std::to_string(err_code));
    }
}

// �p���غc�l (�{�b�u���̰򥻪���l�ơA������ OpenCL �޿�)
opencl_runtime::opencl_runtime() : is_initialized_flag(false), err(CL_SUCCESS) {
    // �o�̤����� OpenCL ��l���޿�
    // �Ҧ��� OpenCL �]�m������ initialize() ��k��
}

// �p���Ѻc�l
opencl_runtime::~opencl_runtime() {
    // cl::Context, cl::CommandQueue, cl::Program ������ RAII �S�ʷ|�۰ʲM�z�귽
    // �o�̵L���㦡�ե� clRelease* ���
}

// �㦡��l�Ƥ�k
void opencl_runtime::initialize() {
    if (is_initialized_flag) {
        std::cout << "opencl_runtime �w�g��l�ơC" << std::endl;
        return;
    }

    try {
        // 1. ������x
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        check_error(err, "cl::Platform::get");
        if (platforms.empty()) {
            throw std::runtime_error("�䤣����� OpenCL ���x�C");
        }
        platform = platforms[0]; // ��ܲĤ@�ӥ��x

        // 2. ����]�� (�u�� GPU)
        std::vector<cl::Device> devices_found;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_found);
        check_error(err, "platform.getDevices (GPU)");
        if (devices_found.empty()) {
            std::cerr << "ĵ�i�G�䤣�� GPU �]�ơA���մM���L OpenCL �]��..." << std::endl;
            err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_found);
            check_error(err, "platform.getDevices (ALL)");
        }
        if (devices_found.empty()) {
            throw std::runtime_error("�䤣����� OpenCL �]�ơC");
        }
        device = devices_found[0]; // ��ܲĤ@�ӧ�쪺�]��
        std::cout << "��� OpenCL �]��: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // 3. �ЫؤW�U��
        context = cl::Context(device, nullptr, nullptr, nullptr, &err);
        check_error(err, "cl::Context constructor");

        // 4. �ЫةR�O���C
        queue = cl::CommandQueue(context, device, 0, &err); // 0 �N��w�]�ݩ�
        check_error(err, "cl::CommandQueue constructor");

        // 5. Ū���ýsĶ Kernel �{��
        std::string kernel_source = get_kernel_file_content(kernel_file_name);
        cl::Program::Sources sources(1, std::make_pair(kernel_source.c_str(), kernel_source.length()));
        program = cl::Program(context, sources, &err);
        check_error(err, "cl::Program constructor");

        // 6. �sĶ�{��
        err = program.build(devices_found); // �w��Ҧ���쪺�]�ƽsĶ
        if (err != CL_SUCCESS) {
            std::string build_log;
            // �T�O����z��ڽsĶ���]�ƪ���x (�Ҧp devices_found[0])
            program.getBuildInfo(devices_found[0], CL_PROGRAM_BUILD_LOG, &build_log);
            std::cerr << "OPENCL KERNEL �{���sĶ����!\n";
            std::cerr << "�]�� (" << devices_found[0].getInfo<CL_DEVICE_NAME>() << ") ���sĶ��x:\n" << build_log << std::endl;
            check_error(err, "program.build"); // �o�|�ߥX���`
        }

        is_initialized_flag = true; // �аO���w���\��l��
        std::cout << "OpenCL �B��ɪ�l�Ʀ��\�C" << std::endl;

    }
    catch (const std::runtime_error& e) {
        std::cerr << "�B��ɿ��~�o�ͦb opencl_runtime ��l�Ƥ�: " << e.what() << std::endl;
        is_initialized_flag = false; // �аO����l�ƥ���
        throw; // ���s�ߥX�A�� main ��Ʈ���
    }
    catch (const std::exception& e) {
        std::cerr << "�q�ο��~�o�ͦb opencl_runtime ��l�Ƥ�: " << e.what() << std::endl;
        is_initialized_flag = false; // �аO����l�ƥ���
        throw std::runtime_error("opencl_runtime ��l�ƥ��ѡG���w�����~�C"); // ���s�ߥX���зǲ��`
    }
}

// Getter ��k�A�{�b�|�ˬd�O�_�w��l��
cl::Context& opencl_runtime::get_context() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime ����l�ơC�Х��I�s initialize()�C");
    }
    return context;
}

cl::Program& opencl_runtime::get_program() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime ����l�ơC�Х��I�s initialize()�C");
    }
    return program;
}

cl::CommandQueue& opencl_runtime::get_queue() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime ����l�ơC�Х��I�s initialize()�C");
    }
    return queue;
}

cl::Device& opencl_runtime::get_device() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime ����l�ơC�Х��I�s initialize()�C");
    }
    return device;
}