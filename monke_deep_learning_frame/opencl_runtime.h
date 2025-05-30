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
    cl::Device device; // 儲存選定的設備
    cl::Program program;
    cl_int err; // 保留 cl_int 錯誤碼檢查

    bool is_initialized_flag; // 追蹤是否已成功初始化

    // 私有建構子，防止外部直接創建實例
    opencl_runtime();

    // 私有解構子，用於單例的清理
    ~opencl_runtime();

    // 禁用拷貝建構子和拷貝賦值運算子，確保單例唯一性
    opencl_runtime(const opencl_runtime&) = delete;
    opencl_runtime& operator=(const opencl_runtime&) = delete;

    // 輔助函式：檢查 OpenCL 錯誤並退出
    void check_error(cl_int err, const char* operation);

public:
    // 獲取單例實例
    static opencl_runtime& getInstance();

    // **新增：顯式初始化方法**
    void initialize();

    // 獲取上下文
    cl::Context& get_context();

    // 獲取程式
    cl::Program& get_program();

    // 獲取命令隊列
    cl::CommandQueue& get_queue();

    // 獲取設備
    cl::Device& get_device();
};