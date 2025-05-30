#include "opencl_runtime.h"
#include <stdexcept> // for std::runtime_error

// 靜態常量，Kernel 文件名
static const std::string kernel_file_name = "kernel.cl";

// 輔助函式：讀取 Kernel 文件內容
std::string get_kernel_file_content(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << file_name << std::endl;
        throw std::runtime_error("無法打開 Kernel 文件: " + file_name);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    std::string content = buffer.str();
    if (content.empty()) {
        std::cerr << "Error: Kernel file is empty: " << file_name << std::endl;
        throw std::runtime_error("Kernel 文件為空: " + file_name);
    }
    return content;
}

// 單例實例的獲取
opencl_runtime& opencl_runtime::getInstance() {
    static opencl_runtime instance;
    return instance;
}

// 輔助函式：檢查 OpenCL 錯誤並拋出異常
void opencl_runtime::check_error(cl_int err_code, const char* operation) {
    if (err_code != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err_code << std::endl;
        // 將 cl_int 錯誤轉換為 std::runtime_error 拋出，以便在 main 函數中統一捕獲
        throw std::runtime_error(std::string("OpenCL Error: ") + operation + " failed with code " + std::to_string(err_code));
    }
}

// 私有建構子 (現在只做最基本的初始化，不執行 OpenCL 邏輯)
opencl_runtime::opencl_runtime() : is_initialized_flag(false), err(CL_SUCCESS) {
    // 這裡不執行 OpenCL 初始化邏輯
    // 所有的 OpenCL 設置都移到 initialize() 方法中
}

// 私有解構子
opencl_runtime::~opencl_runtime() {
    // cl::Context, cl::CommandQueue, cl::Program 等物件的 RAII 特性會自動清理資源
    // 這裡無需顯式調用 clRelease* 函數
}

// 顯式初始化方法
void opencl_runtime::initialize() {
    if (is_initialized_flag) {
        std::cout << "opencl_runtime 已經初始化。" << std::endl;
        return;
    }

    try {
        // 1. 獲取平台
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        check_error(err, "cl::Platform::get");
        if (platforms.empty()) {
            throw std::runtime_error("找不到任何 OpenCL 平台。");
        }
        platform = platforms[0]; // 選擇第一個平台

        // 2. 獲取設備 (優先 GPU)
        std::vector<cl::Device> devices_found;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_found);
        check_error(err, "platform.getDevices (GPU)");
        if (devices_found.empty()) {
            std::cerr << "警告：找不到 GPU 設備，嘗試尋找其他 OpenCL 設備..." << std::endl;
            err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_found);
            check_error(err, "platform.getDevices (ALL)");
        }
        if (devices_found.empty()) {
            throw std::runtime_error("找不到任何 OpenCL 設備。");
        }
        device = devices_found[0]; // 選擇第一個找到的設備
        std::cout << "找到 OpenCL 設備: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // 3. 創建上下文
        context = cl::Context(device, nullptr, nullptr, nullptr, &err);
        check_error(err, "cl::Context constructor");

        // 4. 創建命令隊列
        queue = cl::CommandQueue(context, device, 0, &err); // 0 代表預設屬性
        check_error(err, "cl::CommandQueue constructor");

        // 5. 讀取並編譯 Kernel 程式
        std::string kernel_source = get_kernel_file_content(kernel_file_name);
        cl::Program::Sources sources(1, std::make_pair(kernel_source.c_str(), kernel_source.length()));
        program = cl::Program(context, sources, &err);
        check_error(err, "cl::Program constructor");

        // 6. 編譯程式
        err = program.build(devices_found); // 針對所有找到的設備編譯
        if (err != CL_SUCCESS) {
            std::string build_log;
            // 確保獲取您實際編譯的設備的日誌 (例如 devices_found[0])
            program.getBuildInfo(devices_found[0], CL_PROGRAM_BUILD_LOG, &build_log);
            std::cerr << "OPENCL KERNEL 程式編譯失敗!\n";
            std::cerr << "設備 (" << devices_found[0].getInfo<CL_DEVICE_NAME>() << ") 的編譯日誌:\n" << build_log << std::endl;
            check_error(err, "program.build"); // 這會拋出異常
        }

        is_initialized_flag = true; // 標記為已成功初始化
        std::cout << "OpenCL 運行時初始化成功。" << std::endl;

    }
    catch (const std::runtime_error& e) {
        std::cerr << "運行時錯誤發生在 opencl_runtime 初始化中: " << e.what() << std::endl;
        is_initialized_flag = false; // 標記為初始化失敗
        throw; // 重新拋出，讓 main 函數捕獲
    }
    catch (const std::exception& e) {
        std::cerr << "通用錯誤發生在 opencl_runtime 初始化中: " << e.what() << std::endl;
        is_initialized_flag = false; // 標記為初始化失敗
        throw std::runtime_error("opencl_runtime 初始化失敗：未預期錯誤。"); // 重新拋出為標準異常
    }
}

// Getter 方法，現在會檢查是否已初始化
cl::Context& opencl_runtime::get_context() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime 未初始化。請先呼叫 initialize()。");
    }
    return context;
}

cl::Program& opencl_runtime::get_program() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime 未初始化。請先呼叫 initialize()。");
    }
    return program;
}

cl::CommandQueue& opencl_runtime::get_queue() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime 未初始化。請先呼叫 initialize()。");
    }
    return queue;
}

cl::Device& opencl_runtime::get_device() {
    if (!is_initialized_flag) {
        throw std::runtime_error("opencl_runtime 未初始化。請先呼叫 initialize()。");
    }
    return device;
}